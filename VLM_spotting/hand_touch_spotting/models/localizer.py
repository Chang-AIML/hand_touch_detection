"""Localizer — the LLM outputs the frame index as ordinary digit tokens.

Per sample, in embedding space:
    [instr] [ViT anchor(sec0)] [idx "0"][motion_0] .. [idx "f"][motion_f] [anchor(sec1)] ..
            [" " + question + " Answer with the single frame index. Answer:"]
            [" " + answer + EOS]            <- answer chunk: training only
  ViT anchor(sec)  frozen Qwen-ViT tokens for that second (no index handle)
  idx "f"          real digit tokens (frozen embeds) — the index HANDLE the LLM copies
  motion_f         FrameCompress(grid)[f]  (the trainable connector, one token per frame)
Training: teacher-forced CE on the answer digits + EOS (labels=-100 elsewhere).
Inference: greedy-generate the index list (stops at EOS) -> parse digit runs -> frames.
"""
from __future__ import annotations
import math
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Localizer(nn.Module):
    def __init__(self, W, compress, use_anchor: bool = True, anchor_max_side: int = 168,
                 anchor_stride: int = 1, fps: int = 15, max_frames: int = 320,
                 grad_checkpoint: bool = True):
        super().__init__()
        self.W = W
        self.compress = compress                          # the trainable FrameCompress connector
        self.use_anchor = use_anchor
        self.anchor_max_side = anchor_max_side
        self.anchor_stride = anchor_stride                # place a ViT anchor every N seconds (1 = 1fps)
        self.fps = fps
        self.device = W.device
        self.embed = W.embed_tokens                       # frozen nn.Embedding
        self.eos_id = W.tokenizer.eos_token_id
        assert self.eos_id is not None
        tok = W.tokenizer
        # INVARIANT: these two literals are verbatim — any change shifts tokenization and mAP.
        self._instr_ids = tok("Below are per-frame motion tokens, each preceded by its frame index.",
                              add_special_tokens=False).input_ids
        self._ans_tag_ids = tok(" Answer with the single frame index. Answer:",
                                add_special_tokens=False).input_ids
        # index handle: NO leading space (the motion token separates consecutive indices) -> fewer tokens
        self.idx_ids = [tok(str(f), add_special_tokens=False).input_ids for f in range(max_frames)]
        if grad_checkpoint:
            # non-reentrant is mandatory under FSDP (reentrant leaks memory + fights unshard/reshard hooks)
            W.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            W.model.config.use_cache = False

    NEG_ANSWER = "there is no frame related to the action"   # INVARIANT: verbatim (empty-target text)

    def _emb_ids(self, ids: List[int]) -> torch.Tensor:
        return self.embed(torch.tensor(ids, device=self.device, dtype=torch.long))

    @staticmethod
    def _answer_str(s):
        """Target text: comma-separated GLOBAL event frames, or the NL negative if the window is empty."""
        fr = sorted(int(x) for x in s["event_frames"])
        return ", ".join(str(x) for x in fr) if fr else Localizer.NEG_ANSWER

    @torch.no_grad()
    def _anchors_for_secs(self, secs, frames, fps):
        """Frozen ViT anchor tokens per second, sampled window-locally at sec*fps from the window
        `frames` (T,H,W,3 uint8). Returns {sec: anchor_tensor (g_s, d)}."""
        T = int(frames.shape[0])
        imgs = [np.asarray(frames[min(int(s) * fps, T - 1)]) for s in secs]
        groups = self.W.vit_anchor_groups(imgs, max_side=self.anchor_max_side)
        return {s: g.to(self.device, torch.bfloat16) for s, g in zip(secs, groups)}

    def _build(self, s, with_answer: bool):
        """Assemble one sample's (embeds [S,d], labels [S]); labels=-100 except the answer chunk."""
        fps, N = s["fps"], s["num_frames"]
        motion = self.compress(s["grid"], None, N).to(torch.bfloat16)     # (N, d_llm); text_emb unused
        num_secs = s.get("anchor_num_secs", math.ceil(N / fps))
        start_sec = s.get("anchor_start_sec", 0)
        used = [sec for sec in range(num_secs) if sec % self.anchor_stride == 0]
        anchors = (self._anchors_for_secs([start_sec + sec for sec in used], s["frames"], fps)
                   if self.use_anchor else {})
        segs, labels = [], []

        def add(emb, lab):
            segs.append(emb); labels.extend(lab)

        add(self._emb_ids(self._instr_ids), [-100] * len(self._instr_ids))
        for sec in range(num_secs):
            gsec = start_sec + sec
            if self.use_anchor and gsec in anchors:
                g = anchors[gsec]; add(g, [-100] * g.shape[0])
            for f in range(sec * fps, min((sec + 1) * fps, N)):
                ids = self.idx_ids[f]
                add(self._emb_ids(ids), [-100] * len(ids))        # index handle (the LLM copies it)
                add(motion[f:f + 1], [-100])                      # the motion token
        q_ids = self.W.tokenizer(" " + s["question"], add_special_tokens=False).input_ids
        add(self._emb_ids(q_ids + self._ans_tag_ids), [-100] * (len(q_ids) + len(self._ans_tag_ids)))
        if with_answer:
            a_ids = self.W.tokenizer(" " + self._answer_str(s),
                                     add_special_tokens=False).input_ids + [self.eos_id]
            add(self._emb_ids(a_ids), list(a_ids))                # teacher-forced target
        return torch.cat(segs, 0), torch.tensor(labels, device=self.device, dtype=torch.long)

    def loss_batch(self, batch):
        """Batched teacher-forced CE over the answer tokens."""
        built = [self._build(s, with_answer=True) for s in batch]
        seqs = [e for e, _ in built]
        B, S, d = len(seqs), max(e.shape[0] for e in seqs), seqs[0].shape[-1]
        emb = torch.zeros(B, S, d, device=self.device, dtype=seqs[0].dtype)
        att = torch.zeros(B, S, device=self.device, dtype=torch.long)
        lab = torch.full((B, S), -100, device=self.device, dtype=torch.long)
        for i, (e, l) in enumerate(built):
            n = e.shape[0]; emb[i, :n] = e; att[i, :n] = 1; lab[i, :n] = l
        return self.W.model(inputs_embeds=emb, attention_mask=att, labels=lab,
                            position_ids=None, use_cache=False).loss

    def _left_pad(self, batch):
        """Build + LEFT-pad a batch for generation. Returns (emb, att)."""
        seqs = [self._build(s, with_answer=False)[0] for s in batch]
        B, S, d = len(seqs), max(e.shape[0] for e in seqs), seqs[0].shape[-1]
        emb = torch.zeros(B, S, d, device=self.device, dtype=seqs[0].dtype)
        att = torch.zeros(B, S, device=self.device, dtype=torch.long)
        for i, e in enumerate(seqs):
            n = e.shape[0]; emb[i, S - n:] = e; att[i, S - n:] = 1
        return emb, att

    @staticmethod
    def _parse(recs, decode):
        """recs: list of (token_id, prob) -> ([frames], [scores]). Group consecutive digit chars into
        an integer; score = mean per-digit probability."""
        frames, scores, buf, bufp = [], [], "", []
        for tid, p in recs:
            ch = decode([tid]).strip()
            if ch.isdigit():
                buf += ch; bufp.append(p)
            elif buf:
                frames.append(int(buf)); scores.append(sum(bufp) / len(bufp)); buf, bufp = "", []
        if buf:
            frames.append(int(buf)); scores.append(sum(bufp) / len(bufp))
        return frames, scores

    def _decode_scores(self, gen, n):
        """Shared parse of a generate() result -> list of (frames, scores) for n sequences."""
        seqids, T = gen.sequences, len(gen.scores)
        out = []
        for b in range(n):
            toks = seqids[b].tolist()[-T:]                        # align to the T score steps
            recs = []
            for t in range(T):
                tid = toks[t]
                if tid == self.eos_id:
                    break
                recs.append((tid, float(F.softmax(gen.scores[t][b].float(), dim=-1)[tid])))
            out.append(self._parse(recs, self.W.tokenizer.decode))
        return out

    @torch.no_grad()
    def predict_multievent_batch(self, batch, max_new_tokens: int = 48):
        """Greedy-generate the index list per sample -> list of (frames, scores)."""
        emb, att = self._left_pad(batch)
        gen = self.W.model.generate(inputs_embeds=emb, attention_mask=att, position_ids=None,
                                    max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                                    eos_token_id=self.eos_id, pad_token_id=self.eos_id,
                                    return_dict_in_generate=True, output_scores=True)
        return self._decode_scores(gen, len(batch))

    @torch.no_grad()
    def sample_rollouts(self, s, G: int = 8, temperature: float = 1.0, max_new_tokens: int = 32):
        """Sample G temperature rollouts for ONE sample (GRPO rollout-diversity diagnosis)."""
        emb, att = self._left_pad([s])
        gen = self.W.model.generate(inputs_embeds=emb, attention_mask=att, position_ids=None,
                                    max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                                    top_p=1.0, top_k=0, num_return_sequences=G, eos_token_id=self.eos_id,
                                    pad_token_id=self.eos_id, return_dict_in_generate=True, output_scores=True)
        return self._decode_scores(gen, G)
