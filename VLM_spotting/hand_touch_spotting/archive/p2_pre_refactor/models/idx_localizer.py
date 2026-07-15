"""idx-decode localizer — the LLM OUTPUTS the frame index (as normal digit tokens).

Sequence (per sample), all in embedding space, following the §3.3 interleave with an
index handle placed BEFORE each V-JEPA motion token:

    [instr] [ViT anchor(sec0)] [" 0"][<M_0>] .. [" f"][<M_f>] [ViT anchor(sec1)] ..
            [ " " + question + " Answer:"] [ " {gt}" + EOS ]      <- last chunk: training only

  ViT anchor(sec)  = frozen Qwen-ViT 1-fps tokens (plain, NO index handle)
  " f"             = real digit tokens (frozen embeds) -> the index HANDLE to COPY
  <M_f>            = adaptor(vjepa[f])   (trainable, 1 token/frame)
Training: teacher-forced CE on the answer digits + EOS (HF `labels`, -100 elsewhere).
Inference: greedy generate the index (stops at EOS) -> parse int -> frame index.

use_idx=False drops the " f" handles (ablation: COUNT instead of COPY).
use_anchor=False drops the ViT anchors. lora=True unfreezes all layers with LoRA.
"""
from __future__ import annotations

import glob
import math
import os
import re
from typing import List

import numpy as np
import torch
import torch.nn as nn


class IdxLocalizer(nn.Module):
    def __init__(self, W, adaptor, use_idx: bool = True, use_anchor: bool = True,
                 anchor_max_side: int = 168, anchor_stride: int = 1, fps: int = 15,
                 max_frames: int = 320, grad_checkpoint: bool = True, use_mrope: bool = False,
                 compress=None):
        super().__init__()
        self.W = W
        self.ad = adaptor
        self.compress = compress          # language-injecting grid compressor (None -> mean-pool feats)
        self.use_idx = use_idx
        self.use_anchor = use_anchor
        self.anchor_max_side = anchor_max_side
        self.anchor_stride = anchor_stride    # place a ViT anchor every N seconds (1 = 1fps)
        # M-RoPE: 3-channel position_ids on a VIDEO-TIME clock so the LLM relates
        # motion tokens by TRUE frame distance (motion f & its idx share time=f;
        # anchors at their second w/ 2D spatial grid). Off -> plain sequential RoPE.
        self.use_mrope = use_mrope
        self.fps = fps
        self.device = W.device
        tok = W.tokenizer
        self.embed = W.embed_tokens                       # frozen nn.Embedding
        self.eos_id = tok.eos_token_id
        assert self.eos_id is not None
        instr = ("Below are per-frame motion tokens, each preceded by its frame index."
                 if use_idx else "Below are per-frame motion tokens.")
        self._instr_ids = tok(instr, add_special_tokens=False).input_ids
        self._ans_tag_ids = tok(" Answer with the single frame index. Answer:",
                                add_special_tokens=False).input_ids
        # index handle: NO leading space (the motion token separates consecutive
        # indices, so there's no digit-run ambiguity) -> fewer tokens
        self.idx_ids = [tok(str(f), add_special_tokens=False).input_ids
                        for f in range(max_frames)]
        self._anchor_cache = {}                           # video_id -> list[(g_s,d)] on CPU
        if grad_checkpoint:
            # non-reentrant is mandatory under FSDP (reentrant leaks memory in backward and
            # fights FSDP's unshard/reshard hooks); also fine for the plain-DDP path.
            W.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            W.model.config.use_cache = False

    # ---- embeddings ----
    def _emb_ids(self, ids: List[int]) -> torch.Tensor:
        t = torch.tensor(ids, device=self.device, dtype=torch.long)
        return self.embed(t)

    # ---- ViT anchors (frozen; compute/cache ONLY the used seconds) ----
    @torch.no_grad()
    def _anchors_for_secs(self, video_id, secs, frames=None, fps=None):
        """Return {sec: (anchor_tensor, gh, gw)}. If `frames` (a T,H,W,3 uint8 WINDOW) is given,
        sample the anchor frame WINDOW-LOCALLY at sec*fps from it — no jpg glob, no cache (windows
        aren't revisited). Otherwise fall back to the original cached loose-jpg path."""
        from PIL import Image
        fps = fps or self.fps
        if frames is not None:                                  # windowed: anchors from window frames
            T = int(frames.shape[0])
            imgs = [np.asarray(frames[min(int(s) * fps, T - 1)]) for s in secs]
            groups, grids = self.W.vit_anchor_groups(imgs, max_side=self.anchor_max_side, return_grid=True)
            return {s: (g.to(self.device, torch.bfloat16), gh, gw)
                    for s, (g, (gh, gw)) in zip(secs, zip(groups, grids))}
        cache = self._anchor_cache.setdefault(video_id, {})
        need = [s for s in secs if s not in cache]
        if need:
            fdir = os.environ.get("TOUCH_FRAMES_DIR",
                                  "/home/chang_noroot/data2/huyanh/Workspace/dataset/hoi4d/frames")
            paths = sorted(glob.glob(os.path.join(fdir, video_id, "*.jpg")))
            gframes = [np.asarray(Image.open(paths[min(s * fps, len(paths) - 1)]).convert("RGB"))
                       for s in need]
            groups, grids = self.W.vit_anchor_groups(gframes, max_side=self.anchor_max_side,
                                                     return_grid=True)
            for s, g, (gh, gw) in zip(need, groups, grids):
                cache[s] = (g.to("cpu"), gh, gw)
        return {s: (cache[s][0].to(self.device, torch.bfloat16), cache[s][1], cache[s][2])
                for s in secs}

    # negative answer: natural-language "not present" instead of the bare token 'none'.
    # Parses to zero frames (no digits) and is better aligned with the frozen LLM's language
    # space; also discourages the degenerate "answer 'none' for everything" collapse.
    NEG_ANSWER = "there is no frame related to the action"

    @staticmethod
    def _answer_str(s):
        """Target text. Multi-event (event_frames present): comma list of GLOBAL frames,
        the NL negative if empty. Single-event: the one gt index."""
        if s.get("event_frames") is not None:
            fr = sorted(int(x) for x in s["event_frames"])
            return ", ".join(str(x) for x in fr) if fr else IdxLocalizer.NEG_ANSWER
        return str(int(s["gt"]))

    # ---- feature extraction: per-frame F1 (pre-LLM adaptor) + F2 (post-LLM hidden) ----
    @torch.no_grad()
    def extract_frame_features(self, batch, layer=None):
        """batch of full-video samples (same layout) -> per sample (F1 [N,d], F2 [N,d]):
        F1 = adaptor output (motion tokens as fed to LLM); F2 = LLM hidden at the
        motion-token positions (language-space). layer=None -> last layer; else
        hidden_states[layer] (e.g. 24, 12) for a less over-processed language-space."""
        embs, mpos_all, motions = [], [], []
        for s in batch:
            feats = s["feats"]; N = feats.shape[0]; fps = s["fps"]
            motion = self.ad(feats.to(self.device, torch.bfloat16))       # [N,d] = F1
            num_secs = s.get("anchor_num_secs", math.ceil(N / fps))
            start_sec = s.get("anchor_start_sec", 0)
            used = [sec for sec in range(num_secs) if sec % self.anchor_stride == 0]
            amap = (self._anchors_for_secs(s["video_id"], [start_sec + sec for sec in used])
                    if self.use_anchor else {})
            segs, mpos, pos = [], [], 0

            def add(emb):
                nonlocal pos
                segs.append(emb); pos += emb.shape[0]

            add(self._emb_ids(self._instr_ids))
            for sec in range(num_secs):
                gsec = start_sec + sec
                if self.use_anchor and gsec in amap:
                    add(amap[gsec][0])
                for f in range(sec * fps, min((sec + 1) * fps, N)):
                    if self.use_idx:
                        add(self._emb_ids(self.idx_ids[f]))
                    mpos.append(pos); add(motion[f:f + 1])
            q_ids = self.W.tokenizer(" " + s["question"], add_special_tokens=False).input_ids
            add(self._emb_ids(q_ids + self._ans_tag_ids))
            embs.append(torch.cat(segs, 0)); mpos_all.append(mpos); motions.append(motion.float().cpu())
        S = max(e.shape[0] for e in embs); B = len(embs); d = embs[0].shape[-1]
        emb = torch.zeros(B, S, d, device=self.device, dtype=embs[0].dtype)
        att = torch.zeros(B, S, device=self.device, dtype=torch.long)
        for i, e in enumerate(embs):
            emb[i, :e.shape[0]] = e; att[i, :e.shape[0]] = 1
        o = self.W.llm(inputs_embeds=emb, attention_mask=att, use_cache=False,
                       output_hidden_states=(layer is not None), return_dict=True)
        h = o.last_hidden_state if layer is None else o.hidden_states[layer]   # [B,S,d]
        out = []
        for i in range(B):
            F2 = h[i, torch.tensor(mpos_all[i], device=self.device)].float().cpu()
            out.append((motions[i], F2))
        return out

    # ---- build one sample's (embeds, labels, positions) ----
    def _build(self, s, with_answer: bool):
        fps = s["fps"]
        if self.compress is not None:                              # LANGUAGE-INJECTION front-end
            N = s["num_frames"]
            q_ids = self.W.tokenizer(" " + s["question"], add_special_tokens=False).input_ids
            text_emb = self.embed(torch.tensor(q_ids, device=self.device))   # (T, d_llm) frozen
            motion = self.compress(s["grid"], text_emb, N).to(torch.bfloat16)  # (N, d_llm)
        else:
            feats = s["feats"]
            N = feats.shape[0]
            motion = self.ad(feats.to(self.device, torch.bfloat16))    # (N,d) trainable
        num_secs = s.get("anchor_num_secs", math.ceil(N / fps))
        start_sec = s.get("anchor_start_sec", 0)
        used_local = [sec for sec in range(num_secs) if sec % self.anchor_stride == 0]
        anchor_map = (self._anchors_for_secs(s["video_id"], [start_sec + sec for sec in used_local],
                                             frames=s.get("frames"), fps=fps)
                      if self.use_anchor else {})
        mr = self.use_mrope
        segs, labels, prows = [], [], []           # prows: [t,h,w] per token (M-RoPE)

        def add(emb, lab, pos=None):
            segs.append(emb)
            labels.extend(lab)
            if mr:
                prows.extend(pos)

        I = len(self._instr_ids)
        add(self._emb_ids(self._instr_ids), [-100] * I,
            [[k, k, k] for k in range(I)] if mr else None)     # instr text: 0..I-1
        base_v = I                                              # video region starts here
        for sec in range(num_secs):
            gsec = start_sec + sec
            if self.use_anchor and gsec in anchor_map:
                g, gh, gw = anchor_map[gsec]
                ba = base_v + gsec * fps                        # anchor temporal = its frame time
                apos = ([[ba, ba + (k // gw), ba + (k % gw)] for k in range(g.shape[0])]
                        if mr else None)                        # t const, h/w = 2D grid
                add(g, [-100] * g.shape[0], apos)
            for f in range(sec * fps, min((sec + 1) * fps, N)):
                ft = base_v + f                                 # motion & its idx share time=f
                if self.use_idx:
                    ids = self.idx_ids[f]
                    add(self._emb_ids(ids), [-100] * len(ids), [[ft, ft, ft]] * len(ids) if mr else None)
                add(motion[f:f + 1], [-100], [[ft, ft, ft]] if mr else None)
        base_t = base_v + N                                     # text after the video
        q_ids = self.W.tokenizer(" " + s["question"], add_special_tokens=False).input_ids
        qn = len(q_ids) + len(self._ans_tag_ids)
        add(self._emb_ids(q_ids + self._ans_tag_ids), [-100] * qn,
            [[base_t + k] * 3 for k in range(qn)] if mr else None)
        base_t += qn
        if with_answer:
            a_ids = self.W.tokenizer(" " + self._answer_str(s),
                                     add_special_tokens=False).input_ids
            a_ids = a_ids + [self.eos_id]
            add(self._emb_ids(a_ids), list(a_ids),
                [[base_t + k] * 3 for k in range(len(a_ids))] if mr else None)
        embeds = torch.cat(segs, dim=0)
        lab = torch.tensor(labels, device=self.device, dtype=torch.long)
        pos = (torch.tensor(prows, dtype=torch.long, device=self.device).t().contiguous()
               if mr else None)                                 # [3, S]
        return embeds, lab, pos

    # ---- training: batched teacher-forced CE ----
    def loss_batch(self, batch):
        seqs, labs, poss = zip(*(self._build(s, with_answer=True) for s in batch))
        S = max(e.shape[0] for e in seqs)
        d = seqs[0].shape[-1]
        B = len(seqs)
        emb = torch.zeros(B, S, d, device=self.device, dtype=seqs[0].dtype)
        att = torch.zeros(B, S, device=self.device, dtype=torch.long)
        lab = torch.full((B, S), -100, device=self.device, dtype=torch.long)
        pos = torch.zeros(3, B, S, device=self.device, dtype=torch.long) if self.use_mrope else None
        for i, (e, l, p) in enumerate(zip(seqs, labs, poss)):
            n = e.shape[0]
            emb[i, :n] = e; att[i, :n] = 1; lab[i, :n] = l
            if self.use_mrope:
                pos[:, i, :n] = p
        out = self.W.model(inputs_embeds=emb, attention_mask=att, labels=lab,
                           position_ids=pos, use_cache=False)
        return out.loss

    def _left_pad(self, batch):
        """Build + LEFT-pad a batch for generation. Returns (emb, att, pos or None)."""
        built = [self._build(s, with_answer=False) for s in batch]
        seqs = [b[0] for b in built]
        S = max(e.shape[0] for e in seqs)
        d = seqs[0].shape[-1]
        B = len(seqs)
        emb = torch.zeros(B, S, d, device=self.device, dtype=seqs[0].dtype)
        att = torch.zeros(B, S, device=self.device, dtype=torch.long)
        pos = torch.zeros(3, B, S, device=self.device, dtype=torch.long) if self.use_mrope else None
        for i, e in enumerate(seqs):
            n = e.shape[0]
            emb[i, S - n:] = e; att[i, S - n:] = 1
            if self.use_mrope:
                pos[:, i, S - n:] = built[i][2]
        return emb, att, pos

    # ---- inference: greedy-generate the index ----
    @torch.no_grad()
    def predict_batch(self, batch, max_new_tokens: int = 6):
        emb, att, pos = self._left_pad(batch)
        gen = self.W.model.generate(inputs_embeds=emb, attention_mask=att, position_ids=pos,
                                    max_new_tokens=max_new_tokens, do_sample=False,
                                    num_beams=1, eos_token_id=self.eos_id,
                                    pad_token_id=self.eos_id)
        texts = self.W.tokenizer.batch_decode(gen, skip_special_tokens=True)
        preds = []
        for t in texts:
            m = re.search(r"-?\d+", t)
            preds.append(int(m.group()) if m else -1)
        return preds, texts

    @staticmethod
    def _parse(recs, decode):
        """recs: list of (token_id, prob). -> ([frames], [scores])."""
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

    # ---- inference: multi-event -> list of (frame, score) via generation likelihood ----
    @torch.no_grad()
    def predict_multievent_batch(self, batch, max_new_tokens: int = 48):
        import torch.nn.functional as F
        if self.use_mrope:
            return self._manual_decode(batch, max_new_tokens)   # correct video-time positions
        emb, att, pos = self._left_pad(batch)
        B = len(batch)
        gen = self.W.model.generate(inputs_embeds=emb, attention_mask=att, position_ids=pos,
                                    max_new_tokens=max_new_tokens, do_sample=False,
                                    num_beams=1, eos_token_id=self.eos_id,
                                    pad_token_id=self.eos_id, return_dict_in_generate=True,
                                    output_scores=True)
        seqids = gen.sequences                       # [B, T] (only new tokens for inputs_embeds)
        T = len(gen.scores)
        out = []
        for b in range(B):
            toks = seqids[b].tolist()[-T:]           # align to the T score steps
            recs = []
            for t in range(T):
                tid = toks[t]
                if tid == self.eos_id:
                    break
                recs.append((tid, float(F.softmax(gen.scores[t][b].float(), dim=-1)[tid])))
            out.append(self._parse(recs, self.W.tokenizer.decode))
        return out

    @torch.no_grad()
    def sample_rollouts(self, s, G: int = 8, temperature: float = 1.0, max_new_tokens: int = 32):
        """Sample G rollouts for ONE sample via temperature sampling (do_sample=True) — for GRPO
        rollout-diversity diagnosis. Returns a list of G parsed (frames, scores). Non-mrope path."""
        import torch.nn.functional as F
        emb, att, pos = self._left_pad([s])                      # batch 1
        gen = self.W.model.generate(inputs_embeds=emb, attention_mask=att, position_ids=pos,
                                    max_new_tokens=max_new_tokens, do_sample=True,
                                    temperature=temperature, top_p=1.0, top_k=0,
                                    num_return_sequences=G, eos_token_id=self.eos_id,
                                    pad_token_id=self.eos_id, return_dict_in_generate=True,
                                    output_scores=True)
        seqids = gen.sequences                                   # [G, T]
        T = len(gen.scores)
        out = []
        for b in range(G):
            toks = seqids[b].tolist()[-T:]
            recs = []
            for t in range(T):
                tid = toks[t]
                if tid == self.eos_id:
                    break
                recs.append((tid, float(F.softmax(gen.scores[t][b].float(), dim=-1)[tid])))
            out.append(self._parse(recs, self.W.tokenizer.decode))
        return out

    # ---- shared-prefix eval: encode the QUESTION-INDEPENDENT window prefix (instr + interleaved
    #      anchor/idx/motion tokens) and its LLM KV ONCE, reuse across ALL type-queries on that
    #      window. Valid only when use_text=0 (FrameCompress ignores the question -> motion tokens
    #      are the same for every query) and use_mrope=0 (the trained config). ~len(questions)x
    #      cheaper on the (expensive) prefix: connector + ViT anchors + LLM attention over ~600 toks.
    def _build_prefix_emb(self, s):
        """Instr + interleaved [anchor, idx, motion] video region, WITHOUT the question. [P, d]."""
        fps = s["fps"]; N = s["num_frames"]
        if self.compress is not None:
            motion = self.compress(s["grid"], None, N).to(torch.bfloat16)   # text_emb unused (use_text=0)
        else:
            motion = self.ad(s["feats"].to(self.device, torch.bfloat16))
        num_secs = s.get("anchor_num_secs", math.ceil(N / fps))
        start_sec = s.get("anchor_start_sec", 0)
        used_local = [sec for sec in range(num_secs) if sec % self.anchor_stride == 0]
        anchor_map = (self._anchors_for_secs(s["video_id"], [start_sec + sec for sec in used_local],
                                             frames=s.get("frames"), fps=fps) if self.use_anchor else {})
        segs = [self._emb_ids(self._instr_ids)]
        for sec in range(num_secs):
            gsec = start_sec + sec
            if self.use_anchor and gsec in anchor_map:
                g, gh, gw = anchor_map[gsec]
                segs.append(g)
            for f in range(sec * fps, min((sec + 1) * fps, N)):
                if self.use_idx:
                    segs.append(self._emb_ids(self.idx_ids[f]))
                segs.append(motion[f:f + 1])
        return torch.cat(segs, dim=0)

    def _expand_past(self, past, Q):
        """Repeat a batch-1 KV cache to batch Q (contiguous: each query appends independently)."""
        def exp(t):
            return t.expand(Q, *t.shape[1:]).contiguous()
        if hasattr(past, "to_legacy_cache") and hasattr(type(past), "from_legacy_cache"):
            legacy = tuple(tuple(exp(t) for t in layer) for layer in past.to_legacy_cache())
            return type(past).from_legacy_cache(legacy)
        if hasattr(past, "key_cache"):
            for i in range(len(past.key_cache)):
                past.key_cache[i] = exp(past.key_cache[i]); past.value_cache[i] = exp(past.value_cache[i])
            return past
        return tuple(tuple(exp(t) for t in layer) for layer in past)

    @torch.no_grad()
    def predict_shared_prefix(self, window, questions, max_new_tokens: int = 64, verify: bool = False):
        """Eval every query in `questions` on ONE `window`, sharing the prefix encode + KV.
        Returns list of (frames, scores) — identical to per-query predict_multievent_batch but
        ~len(questions)x cheaper on the prefix. verify=True asserts equality vs the naive path."""
        import torch.nn.functional as F
        assert not self.use_mrope, "shared-prefix requires use_mrope=0"
        assert self.compress is None or not self.compress.use_text, "shared-prefix requires use_text=0"
        dev = self.device
        pre = self._build_prefix_emb(window)                          # [P, d] — connector + anchors ONCE
        P, d = pre.shape[0], pre.shape[-1]
        o = self.W.model(inputs_embeds=pre[None],
                         attention_mask=torch.ones(1, P, device=dev, dtype=torch.long), use_cache=True)
        past = self._expand_past(o.past_key_values, len(questions))   # prefill KV -> expand to Q
        q_segs = [self._emb_ids(self.W.tokenizer(" " + q, add_special_tokens=False).input_ids
                                + self._ans_tag_ids) for q in questions]
        Q = len(q_segs); Lq = max(e.shape[0] for e in q_segs)
        qemb = torch.zeros(Q, Lq, d, device=dev, dtype=pre.dtype)     # LEFT-pad queries
        qatt = torch.zeros(Q, Lq, device=dev, dtype=torch.long); qn = []
        for i, e in enumerate(q_segs):
            n = e.shape[0]; qemb[i, Lq - n:] = e; qatt[i, Lq - n:] = 1; qn.append(n)
        att = torch.cat([torch.ones(Q, P, device=dev, dtype=torch.long), qatt], dim=1)
        posq = torch.zeros(Q, Lq, device=dev, dtype=torch.long)       # prefix=0..P-1, real query=P..P+n-1
        for i in range(Q):
            posq[i, Lq - qn[i]:] = torch.arange(P, P + qn[i], device=dev)
        o = self.W.model(inputs_embeds=qemb, attention_mask=att, position_ids=posq,
                         past_key_values=past, use_cache=True)
        past = o.past_key_values; logits = o.logits[:, -1, :]
        recs = [[] for _ in range(Q)]; done = [False] * Q
        cur = torch.tensor([P + qn[i] for i in range(Q)], device=dev)
        for step in range(max_new_tokens):
            probs = F.softmax(logits.float(), dim=-1); nxt = logits.argmax(-1)
            pv = probs.gather(1, nxt[:, None])[:, 0]
            for b in range(Q):
                if done[b]:
                    continue
                if int(nxt[b]) == self.eos_id:
                    done[b] = True
                else:
                    recs[b].append((int(nxt[b]), float(pv[b])))
            if all(done):
                break
            att = torch.cat([att, torch.ones(Q, 1, device=dev, dtype=torch.long)], dim=1); cur = cur + 1
            o = self.W.model(inputs_embeds=self.embed(nxt).unsqueeze(1), attention_mask=att,
                             position_ids=cur[:, None], past_key_values=past, use_cache=True)
            past = o.past_key_values; logits = o.logits[:, -1, :]
        out = [self._parse(recs[b], self.W.tokenizer.decode) for b in range(Q)]
        if verify:                                                    # correctness gate vs naive per-query
            naive = self.predict_multievent_batch([{**window, "question": q} for q in questions],
                                                  max_new_tokens=max_new_tokens)
            for b in range(Q):
                assert out[b][0] == naive[b][0], f"shared-prefix MISMATCH q{b}: {out[b][0]} vs {naive[b][0]}"
        return out

    # ---- M-RoPE generation: manual greedy decode so generated answer tokens get the
    #      SAME video-time positions (base_t + step) as teacher-forced training (HF
    #      generate would give them sequential/rope_delta positions -> train/eval MISMATCH) ----
    @torch.no_grad()
    def _manual_decode(self, batch, max_new_tokens: int = 48):
        import torch.nn.functional as F
        dev = self.device
        built = [self._build(s, with_answer=False) for s in batch]
        seqs = [b[0] for b in built]
        B = len(batch)
        S = max(e.shape[0] for e in seqs)
        d = seqs[0].shape[-1]
        emb = torch.zeros(B, S, d, device=dev, dtype=seqs[0].dtype)   # LEFT-pad
        att = torch.zeros(B, S, device=dev, dtype=torch.long)
        pos = torch.zeros(3, B, S, device=dev, dtype=torch.long)
        nextp = torch.zeros(3, B, device=dev, dtype=torch.long)       # base_t per sample
        for i, e in enumerate(seqs):
            n = e.shape[0]
            emb[i, S - n:] = e; att[i, S - n:] = 1
            pos[:, i, S - n:] = built[i][2]
            nextp[:, i] = built[i][2][:, -1] + 1                      # last prompt pos + 1
        o = self.W.model(inputs_embeds=emb, attention_mask=att, position_ids=pos, use_cache=True)
        past = o.past_key_values
        logits = o.logits[:, -1, :]
        recs = [[] for _ in range(B)]
        done = [False] * B
        for step in range(max_new_tokens):
            probs = F.softmax(logits.float(), dim=-1)
            nxt = logits.argmax(dim=-1)                               # [B]
            pv = probs.gather(1, nxt[:, None])[:, 0]
            for b in range(B):
                if done[b]:
                    continue
                if int(nxt[b]) == self.eos_id:
                    done[b] = True
                else:
                    recs[b].append((int(nxt[b]), float(pv[b])))
            if all(done):
                break
            tok_emb = self.embed(nxt).unsqueeze(1)                    # [B,1,d]
            att = torch.cat([att, torch.ones(B, 1, device=dev, dtype=torch.long)], dim=1)
            step_pos = (nextp + step).unsqueeze(-1)                   # [3,B,1] video-time
            o = self.W.model(inputs_embeds=tok_emb, attention_mask=att, position_ids=step_pos,
                             past_key_values=past, use_cache=True)
            past = o.past_key_values
            logits = o.logits[:, -1, :]
        return [self._parse(recs[b], self.W.tokenizer.decode) for b in range(B)]
