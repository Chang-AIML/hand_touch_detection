"""Trainable component (2): [LOC] (and Phase-2 [REJ]) token embeddings.

k learnable query embeddings that read out timestamps: the output hidden at each
[LOC] slot is compared (cosine) against the per-frame V-JEPA hiddens -> s(t) ->
soft-argmax. [REJ] (Phase 2) absorbs unmatched queries. Initialised from the LLM's
input-embedding statistics so they start in-distribution for the frozen backbone.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LocTokens(nn.Module):
    def __init__(self, d_llm: int = 4096, k: int = 1, use_rej: bool = False,
                 init_std: float = 0.02):
        super().__init__()
        self.k = k
        self.use_rej = use_rej
        self.loc = nn.Parameter(torch.randn(k, d_llm) * init_std)
        self.rej = nn.Parameter(torch.randn(1, d_llm) * init_std) if use_rej else None

    @torch.no_grad()
    def init_from_embeddings(self, embed_weight: torch.Tensor):
        """Init LOC/REJ near the mean token embedding (+ small noise), matched scale."""
        rms = embed_weight.float().pow(2).mean(dim=1).sqrt().mean().item()
        for p in ([self.loc] + ([self.rej] if self.rej is not None else [])):
            mean = embed_weight.float().mean(dim=0, keepdim=True).to(p.device)
            noise = torch.randn_like(p) * (rms * 0.05)
            p.copy_((mean + noise).to(p.dtype))
        return rms

    def loc_embeds(self, dtype=None, device=None):
        e = self.loc
        return e.to(dtype=dtype, device=device) if dtype or device else e

    def rej_embed(self, dtype=None, device=None):
        if self.rej is None:
            return None
        return self.rej.to(dtype=dtype, device=device) if dtype or device else self.rej
