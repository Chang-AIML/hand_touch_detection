"""Query-conditioned FiLM modulation of the Motion-Bridge output (per user's spec).

The shared adaptor maps every V-JEPA token into one fixed space z_t = Bridge(vjepa_t),
but motion is entangled (contact / release / approach / object / camera). FiLM lets
the language query q = h_loc modulate the motion tokens before the cosine readout:

    gamma, beta = MLP(q)
    z'_t = LayerNorm(z_t) * (1 + gamma) + beta
    score(t) = cos(q, z'_t) / temp

so a touch query can amplify contact-onset dims and a release query the separation
dims. It is NOT a decoder head — the readout stays plain cosine retrieval. Zero-init
of the last layer makes FiLM start as identity (LayerNorm(z)), so it only departs
from the plain bridge as it learns.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FiLM(nn.Module):
    def __init__(self, d_llm: int = 4096, hidden: int = 1024):
        super().__init__()
        self.ln = nn.LayerNorm(d_llm)
        self.q_ln = nn.LayerNorm(d_llm)
        self.mlp = nn.Sequential(nn.Linear(d_llm, hidden), nn.GELU(),
                                 nn.Linear(hidden, 2 * d_llm))
        nn.init.zeros_(self.mlp[-1].weight)          # start as identity (gamma=beta=0)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, q: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """q: (d,) query (h_loc); z: (N, d) bridge output -> z': (N, d) modulated."""
        gb = self.mlp(self.q_ln(q.float()))          # (2d,)
        gamma, beta = gb.chunk(2, dim=-1)            # (d,), (d,)
        return self.ln(z.float()) * (1.0 + gamma) + beta
