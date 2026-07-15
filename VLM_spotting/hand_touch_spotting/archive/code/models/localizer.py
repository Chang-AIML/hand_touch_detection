"""Localizer: bundles frozen Qwen wrapper + trainable adaptor + [LOC], and turns a
batch of samples into per-sample similarity curves s(t) = cos(h_loc, h_vjepa(t))/temp.

Shared by training (grad on) and evaluation (grad off). Builds each sample's §3.3
embedding sequence, right-pads the batch, runs the truncated grad forward to
sim_layer, then reads the LOC/V-JEPA hiddens.
"""
from __future__ import annotations

import sys
import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.token_layout import assemble_embeds  # noqa: E402


class Localizer(nn.Module):
    def __init__(self, wrapper, adaptor, loc, sim_layer: int, temp: float = 0.07,
                 use_anchor: bool = False, fps: int = 15, grad_checkpoint: bool = False,
                 sim_head=None, align_target: str = "postllm", film=None):
        super().__init__()
        self.W = wrapper
        self.adaptor = adaptor
        self.loc = loc
        self.sim_head = sim_head          # trainable projection readout (LISA-style); None -> raw cosine
        self.film = film                  # query-conditioned FiLM on adaptor output (None -> off)
        self.sim_layer = sim_layer
        self.temp = temp
        self.use_anchor = use_anchor
        self.fps = fps
        self.grad_checkpoint = grad_checkpoint
        # what the LOC hidden aligns AGAINST (the "loc 和谁对齐" knob):
        #   'postllm' -> per-frame V-JEPA hidden after the frozen LLM (contextualised)
        #   'adaptor' -> the adaptor OUTPUT tokens (pre-LLM translated motion), same d_llm
        self.align_target = align_target
        self.anchor_max_side = 0          # >0 -> downscale anchor frames (cheaper ViT)
        self._anchor_cache = {}           # video_id -> list[(g_s,d)] on CPU (frozen ViT, reuse across epochs)

    def _anchors_cached(self, video_id, full_num_frames, start_sec, num_secs):
        """Compute the FULL video's 1-fps ViT anchors once (cached on CPU), then
        return the slice [start_sec : start_sec+num_secs] for a (possibly cropped)
        window. Caching stays valid under random crops because crops are fps-aligned."""
        hit = self._anchor_cache.get(video_id)
        dev = next(self.adaptor.parameters()).device
        dt = next(self.adaptor.parameters()).dtype
        if hit is None:
            groups = self.W.vit_anchor_groups(self._frames_1fps(video_id, full_num_frames),
                                              max_side=self.anchor_max_side)
            hit = [g.to("cpu") for g in groups]
            self._anchor_cache[video_id] = hit
        sl = hit[start_sec:start_sec + num_secs]
        return [g.to(dev, dt) for g in sl]

    def _frames_1fps(self, video_id, num_frames):
        import glob
        import numpy as np
        from PIL import Image
        fdir = os.environ.get(
            "TOUCH_FRAMES_DIR",
            "/home/chang_noroot/data2/huyanh/Workspace/dataset/hoi4d/frames")
        paths = sorted(glob.glob(os.path.join(fdir, video_id, "*.jpg")))
        secs = list(range(0, num_frames, self.fps))
        return [np.asarray(Image.open(paths[min(s, len(paths) - 1)]).convert("RGB")) for s in secs]

    def forward_batch(self, batch: List[dict]):
        """batch: list of dataset dicts. Returns (s_list, meta_list):
        s_list[i]: (N_i,) cos-sim / temp (grad-enabled iff adaptor/loc require grad).
        meta_list[i]: dict with gt, vjepa_pos, loc_pos, video_id, type, num_frames."""
        dev = next(self.adaptor.parameters()).device
        dt = next(self.adaptor.parameters()).dtype
        seqs, metas, vj_es = [], [], []
        loc_e = self.loc.loc_embeds(dt, dev)
        for s in batch:
            feats = s["feats"].to(dev, dt)
            N = feats.shape[0]
            vj_e = self.adaptor(feats)                        # (N,d) grad
            vj_es.append(vj_e)
            with torch.no_grad():
                q = self.W.embed_question(s["question"])
                ag = self._anchors_cached(
                    s["video_id"], s.get("full_num_frames", N),
                    s.get("anchor_start_sec", 0),
                    s.get("anchor_num_secs", (N + s["fps"] - 1) // s["fps"])
                ) if self.use_anchor else None
            lay = assemble_embeds(ag or [], vj_e, q, loc_e, N, s["fps"])
            seqs.append(lay["embeds"])
            metas.append({"gt": s["gt"], "vjepa_pos": lay["vjepa_pos"],
                          "loc_pos": lay["loc_pos"], "video_id": s["video_id"],
                          "type": s["type"], "num_frames": N,
                          "event_frames": s.get("event_frames")})
        # right-pad
        Lmax = max(e.shape[0] for e in seqs)
        d = seqs[0].shape[1]
        B = len(seqs)
        padded = torch.zeros(B, Lmax, d, device=dev, dtype=dt)
        mask = torch.zeros(B, Lmax, dtype=torch.long, device=dev)
        for i, e in enumerate(seqs):
            padded[i, : e.shape[0]] = e
            mask[i, : e.shape[0]] = 1
        hidden = self.W.forward_to_layer(padded, self.sim_layer, attention_mask=mask,
                                         use_checkpoint=self.grad_checkpoint and self.training)
        s_list = []
        for i, m in enumerate(metas):
            h = hidden[i]
            hloc = h[int(m["loc_pos"][0])]
            if self.align_target == "adaptor":
                hvj = vj_es[i]                                 # pre-LLM translated motion (N,d)
            else:                                             # 'postllm'
                hvj = h[m["vjepa_pos"].to(dev)]               # contextualised (N,d)
            if self.film is not None:                         # query-conditioned FiLM on keys
                hvj = self.film(hloc, hvj)
                s = F.cosine_similarity(hloc.float()[None], hvj, dim=-1) / self.temp
            elif self.sim_head is not None:
                s = self.sim_head(hloc, hvj, self.temp)
            else:
                s = F.cosine_similarity(hloc.float()[None], hvj.float(), dim=-1) / self.temp
            s_list.append(s)
        return s_list, metas
