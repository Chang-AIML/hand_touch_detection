"""Language-injecting frame compressor at the V-JEPA -> LLM connection (the thing
we're testing). Per frame: 576 spatial patches -> K learnable tokens, conditioned on
the question, then packed to one LLM motion token. Drop-in for `adaptor(feats)`.

Flow (per frame, all cross-attn -> no 576^2, cheap):
  even grid (N/2,576,768) --linear interp--> (N,576,768)          # interp BEFORE compress
  (optional STAIN)  ① text <- patch : ground the question on this frame's patches
                    ② patch <- text : stain patches with grounded question (+skip)
  ③ K learnable q <- patch : pool to K tokens                      # K=8
  concat(K*768) -> Linear -> d_llm  -> RMS-match to LLM embeds
Ablations via flags: stain on/off, K, use_text (vs pure learned queries).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class XAttn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.n = nn.LayerNorm(d)
        self.a = nn.MultiheadAttention(d, h, batch_first=True)

    def forward(self, q, kv):                                     # q[B,Lq,d] kv[B,Lk,d]
        o, _ = self.a(self.n(q), kv, kv)
        return o


class FrameCompress(nn.Module):
    def __init__(self, d=768, n_q=8, d_llm=4096, n_heads=4, stain=True, use_text=True,
                 gate_lang=False, target_rms: float | None = None):
        super().__init__()
        self.n_q, self.stain, self.use_text, self.gate_lang = n_q, stain, use_text, gate_lang
        self.target_rms = target_rms
        self.q = nn.Parameter(torch.randn(n_q, d) * 0.02)
        self.in_ln = nn.LayerNorm(d)
        if use_text:
            self.tproj = nn.Sequential(nn.Linear(d_llm, d), nn.GELU(), nn.Linear(d, d))
        if stain and use_text:
            self.a1 = XAttn(d, n_heads)                           # text <- patch
            self.a2 = XAttn(d, n_heads)                           # patch <- text
        self.a3 = XAttn(d, n_heads)                               # K queries <- patch
        if use_text:                                             # inject text into the K queries too
            self.qcond = nn.Sequential(nn.Linear(d_llm, d), nn.GELU(), nn.Linear(d, d))
        # gates init 0 -> starts as PURE query-pool (the winner); language only adds if it helps
        self.g_stain = nn.Parameter(torch.zeros(1)) if gate_lang else None
        self.g_qcond = nn.Parameter(torch.zeros(1)) if gate_lang else None
        self.out = nn.Sequential(nn.LayerNorm(n_q * d), nn.Linear(n_q * d, d_llm))

    @torch.no_grad()
    def set_target_rms_from(self, embed_weight):
        self.target_rms = embed_weight.float().pow(2).mean(-1).sqrt().mean().item()
        return self.target_rms

    def _interp(self, even, N):                                  # even (N2,576,768) -> (N,576,768)
        n2, P, D = even.shape
        x = even.permute(1, 2, 0).reshape(1, P * D, n2)          # (1, P*D, N2)
        x = F.interpolate(x, size=N, mode="linear", align_corners=False)
        return x.reshape(P, D, N).permute(2, 0, 1).contiguous()  # (N,576,768)

    def forward(self, even_grid, text_emb, N, chunk=64):
        """even_grid (N2,576,768) fp16/32 ; text_emb (T,d_llm) question word-embeds ; N frames."""
        dev = self.q.device
        grid = self._interp(even_grid.to(dev).float(), N)        # (N,576,768)
        txt = self.tproj(text_emb.to(dev).float()).unsqueeze(0) if self.use_text else None  # (1,T,d)
        outs = []
        for s in range(0, N, chunk):
            p = self.in_ln(grid[s:s + chunk])                    # (b,576,d)
            b = p.shape[0]
            gs = self.g_stain if self.gate_lang else 1.0
            gq = self.g_qcond if self.gate_lang else 1.0
            if self.stain and self.use_text:
                t = txt.expand(b, -1, -1)
                gt = t + self.a1(t, p)                           # ① text <- patch (grounded)
                p = p + gs * self.a2(p, gt)                      # ② patch <- text (stain, gated skip)
            q = self.q.unsqueeze(0).expand(b, -1, -1)            # (b,K,d)
            if self.use_text:
                q = q + gq * self.qcond(text_emb.to(dev).float()).mean(0).view(1, 1, -1)
            z = self.a3(q, p)                                    # ③ K <- patch  (b,K,d)
            y = self.out(z.reshape(b, -1))                       # (b, d_llm)
            outs.append(y)
        y = torch.cat(outs, 0)                                   # (N, d_llm)
        if self.target_rms is not None:
            rms = y.pow(2).mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
            y = y / rms * self.target_rms
        return y
