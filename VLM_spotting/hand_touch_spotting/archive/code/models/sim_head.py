"""Trainable similarity-projection head (LISA-style readout).

Comparing the LOC hidden to the per-frame V-JEPA hiddens by raw cosine in the
4096-d LLM space is a fixed, un-learnable metric. Instead we project both through
a small shared MLP into a matching space and take cosine there — so training can
shape *what* makes a frame match the query. Two separate projections (query vs
key) work best in practice (asymmetric roles).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimHead(nn.Module):
    def __init__(self, d_llm: int = 4096, d_proj: int = 256, hidden: int = 1024):
        super().__init__()
        self.q = nn.Sequential(nn.LayerNorm(d_llm), nn.Linear(d_llm, hidden),
                               nn.GELU(), nn.Linear(hidden, d_proj))
        self.k = nn.Sequential(nn.LayerNorm(d_llm), nn.Linear(d_llm, hidden),
                               nn.GELU(), nn.Linear(hidden, d_proj))

    def forward(self, h_loc: torch.Tensor, h_vjepa: torch.Tensor, temp: float):
        """h_loc: (d,); h_vjepa: (N,d) -> s: (N,) cosine(query,key)/temp."""
        q = F.normalize(self.q(h_loc.float()), dim=-1)          # (d_proj,)
        k = F.normalize(self.k(h_vjepa.float()), dim=-1)        # (N, d_proj)
        return (k @ q) / temp
