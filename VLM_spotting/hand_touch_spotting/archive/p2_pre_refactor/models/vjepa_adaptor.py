"""Trainable component (1): V-JEPA motion-token adaptor  d_vjepa -> d_llm.

Lightweight by design (the backbone is frozen; only this + the LOC/REJ embeddings
train). Optionally rescales its output to a target RMS norm so the injected motion
tokens live at the same scale as Qwen's native input embeddings — important because
the frozen LLM is sensitive to input-embedding magnitude (esp. in Phase 0 where the
adaptor is *untrained* and we still want a fair feasibility read).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VJEPAAdaptor(nn.Module):
    def __init__(self, d_vjepa: int = 768, d_llm: int = 4096, hidden: int = 2048,
                 n_layers: int = 2, target_rms: float | None = None,
                 input_ln: bool = True):
        super().__init__()
        self.target_rms = target_rms
        layers = []
        din = d_vjepa
        if input_ln:
            layers.append(nn.LayerNorm(d_vjepa))
        for i in range(n_layers - 1):
            layers += [nn.Linear(din, hidden), nn.GELU()]
            din = hidden
        layers.append(nn.Linear(din, d_llm))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_vjepa) -> (..., d_llm)."""
        y = self.net(x)
        if self.target_rms is not None:
            rms = y.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
            y = y / rms * self.target_rms
        return y

    @torch.no_grad()
    def set_target_rms_from(self, embed_weight: torch.Tensor):
        """Set target_rms to the mean per-token RMS of an embedding matrix."""
        rms = embed_weight.float().pow(2).mean(dim=-1).sqrt().mean().item()
        self.target_rms = rms
        return rms
