"""Stage-2 refiners: take the ±W window of pre-LLM adaptor motion features around a
coarse <idx>, condition on the (grounded) language hidden, and output a per-frame
SALIENCY over the window. δ = soft-argmax(saliency) -> refined frame = idx + δ.

Two variants (try both):
  RefineTCN      — FiLM(language) + dilated 1D-conv residual layers (MS-TCN flavour)
  RefineASFormer — self-attn + language cross-attn blocks (ASFormer flavour)

Per-frame saliency (not δ-regression): position = where the target is SALIENT, and
the language both makes the target salient AND suppresses non-target events (keeps a
single peak so soft-argmax is safe). Output aligns 1:1 with the L window frames.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_argmax(sal_logits, pad_mask, window):
    """sal_logits [B,L] (pad = -inf) -> (delta [B] in [-W,W], peak_prob [B])."""
    w = F.softmax(sal_logits, dim=-1)                       # [B,L]
    pos = torch.arange(-window, window + 1, device=sal_logits.device, dtype=torch.float32)
    delta = (w * pos).sum(-1)                               # [B]
    peak = w.max(-1).values                                 # [B] confidence
    return delta, peak


# ----------------------------------------------------------------- TCN variant
class _DilatedResidual(nn.Module):
    def __init__(self, d, dilation):
        super().__init__()
        self.conv = nn.Conv1d(d, d, 3, padding=dilation, dilation=dilation)
        self.out = nn.Conv1d(d, d, 1)

    def forward(self, x, m):                                # x [B,d,L], m [B,1,L]
        h = F.relu(self.conv(x))
        h = self.out(h)
        return (x + h) * m


class RefineTCN(nn.Module):
    def __init__(self, d_llm=4096, d=256, window=12, n_layers=6):
        super().__init__()
        self.window = window
        self.in_proj = nn.Linear(d_llm, d)
        self.film = nn.Linear(d_llm, 2 * d)                 # language -> (gamma, beta)
        self.layers = nn.ModuleList([_DilatedResidual(d, 2 ** i) for i in range(n_layers)])
        self.sal = nn.Conv1d(d, 1, 1)

    def forward(self, query, V, pad_mask):
        h = self.in_proj(V)                                 # [B,L,d]
        g, b = self.film(query).chunk(2, dim=-1)            # [B,d] each
        h = g.unsqueeze(1) * h + b.unsqueeze(1)             # FiLM (language conditions motion)
        h = h.transpose(1, 2)                               # [B,d,L]
        m = (~pad_mask).float().unsqueeze(1)                # [B,1,L]
        for lyr in self.layers:
            h = lyr(h, m)
        sal = self.sal(h).squeeze(1)                        # [B,L]
        return sal.masked_fill(pad_mask, -1e4)


# ----------------------------------------------------------------- ASFormer variant
class _ASBlock(nn.Module):
    def __init__(self, d, dilation, n_heads):
        super().__init__()
        self.conv = nn.Conv1d(d, d, 3, padding=dilation, dilation=dilation)
        self.self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)   # frames attend language
        self.norm1 = nn.LayerNorm(d); self.norm2 = nn.LayerNorm(d); self.norm3 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, h, lang, pad_mask):                   # h [B,L,d], lang [B,1,d]
        c = F.relu(self.conv(h.transpose(1, 2))).transpose(1, 2)         # dilated conv FF
        h = self.norm1(h + c)
        a, _ = self.self_attn(h, h, h, key_padding_mask=pad_mask)
        h = self.norm2(h + a)
        x, _ = self.cross_attn(h, lang, lang)               # inject language (semantics)
        h = self.norm3(h + x)
        return h + self.ff(h)


class RefineASFormer(nn.Module):
    def __init__(self, d_llm=4096, d=256, window=12, n_layers=5, n_heads=4):
        super().__init__()
        self.window = window
        self.in_proj = nn.Linear(d_llm, d)
        self.q_proj = nn.Sequential(nn.Linear(d_llm, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.zeros(2 * window + 1, d)); nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([_ASBlock(d, 2 ** i, n_heads) for i in range(n_layers)])
        self.sal = nn.Linear(d, 1)

    def forward(self, query, V, pad_mask):
        h = self.in_proj(V) + self.pos.unsqueeze(0)         # [B,L,d] + rel-pos
        lang = self.q_proj(query).unsqueeze(1)              # [B,1,d]
        for blk in self.blocks:
            h = blk(h, lang, pad_mask)
        sal = self.sal(h).squeeze(-1)                       # [B,L]
        return sal.masked_fill(pad_mask, -1e4)


def build_refiner(head_type, d_llm, d, window, n_layers):
    if head_type == "tcn":
        return RefineTCN(d_llm, d, window, n_layers)
    if head_type == "asformer":
        return RefineASFormer(d_llm, d, window, n_layers)
    raise ValueError(head_type)
