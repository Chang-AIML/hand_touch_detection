"""Action Head — δ-offset regressor via ATTENTION-WEIGHTED POSITION READOUT.

  query   = question last-token LLM hidden (whole-sentence semantic summary)
  key/val = pre-LLM adaptor output V over the ±W window (SHARP per-frame motion)
  δ = Σ(attn_weight · relative_position)

The language query cross-attends to the window's motion tokens; wherever it attends
IS the boundary, so δ is read directly from the attention distribution over relative
positions — no pos-embedding, no MLP regression, no tanh bound. δ is naturally in
[-W, +W]. A single shared head handles all action types (query-conditioned), and
generalizes to semantically similar unseen queries.

Precision = sharpness of the attention. If it attends to one frame -> δ is exact
(@0 rises); if it smears over several -> δ pulls to the middle. Remedies if smeared:
attention temperature or an MLP residual — left out for now (add only if needed).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ActionHead(nn.Module):
    def __init__(self, d_llm: int = 4096, d: int = 256, n_heads: int = 4, window: int = 12):
        super().__init__()
        self.window = window
        self.L = 2 * window + 1
        # question 4096 -> d in one step is too harsh -> 2-layer w/ GELU (修4)
        self.q_proj = nn.Sequential(nn.Linear(d_llm, d), nn.GELU(), nn.Linear(d, d))
        self.k_proj = nn.Linear(d_llm, d)                  # motion: single layer is enough
        self.v_proj = nn.Linear(d_llm, d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)

    def forward(self, query, V, pad_mask):
        """query [B,d_llm]; V [B,L,d_llm] pre-LLM adaptor output; pad_mask [B,L] True=pad.
        Returns delta [B] in frames, guaranteed within [-window, +window]."""
        Q = self.q_proj(query).unsqueeze(1)                # [B,1,d]
        K = self.k_proj(V)                                 # [B,L,d]  no pos-emb
        Vv = self.v_proj(V)
        _, attn_w = self.attn(Q, K, Vv, key_padding_mask=pad_mask,
                              need_weights=True, average_attn_weights=True)   # [B,1,L]
        positions = torch.arange(-self.window, self.window + 1,
                                 device=V.device, dtype=torch.float32)        # [L]
        w = attn_w.squeeze(1).masked_fill(pad_mask, 0.0)   # [B,L] mask pad frames
        w = w / (w.sum(-1, keepdim=True) + 1e-8)           # renormalize
        delta = (w * positions).sum(-1)                    # [B]  in [-W, +W]
        return delta
