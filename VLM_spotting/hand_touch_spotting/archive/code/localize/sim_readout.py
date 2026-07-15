"""Similarity readout: s(t)=sim(h_loc, h_vjepa(t)), soft-argmax, and FWHM helpers.

Used by the Phase-0 diagnostic and by the Phase-1 localizer.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def cosine_curve(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """query: (d,); keys: (T, d) -> s: (T,) cosine similarity, float32."""
    q = query.float()
    k = keys.float()
    return F.cosine_similarity(q[None, :], k, dim=-1)


def novelty_curve(feats: torch.Tensor) -> torch.Tensor:
    """Temporal novelty n(t) = 1 - cos(f(t), f(t+1)); pads last -> (T,)."""
    f = feats.float()
    c = F.cosine_similarity(f[:-1], f[1:], dim=-1)
    n = 1.0 - c
    return torch.cat([n, n[-1:]], dim=0)


def soft_argmax(s: torch.Tensor, temp: float = 1.0) -> float:
    """Expected index under softmax(s/temp)."""
    p = F.softmax(s.float() / temp, dim=-1)
    idx = torch.arange(s.shape[-1], device=s.device, dtype=torch.float32)
    return float((p * idx).sum())


def align_to_gt(s: np.ndarray, gt: int, half: int = 15) -> np.ndarray:
    """Extract window s[gt-half .. gt+half] (len 2*half+1), NaN-padded at edges."""
    T = len(s)
    out = np.full(2 * half + 1, np.nan, dtype=np.float64)
    for j, t in enumerate(range(gt - half, gt + half + 1)):
        if 0 <= t < T:
            out[j] = s[t]
    return out


def compute_fwhm(curve: np.ndarray):
    """FWHM (in frames) of the peak nearest the window centre, + peak offset.

    curve: aligned/averaged 1D signal, index `half` == GT frame. Returns
    (fwhm_frames, peak_offset_frames, peak_value). Baseline = curve min; half-max
    = min + 0.5*(max-min). FWHM = width where curve >= half-max around the peak.
    """
    c = np.asarray(curve, dtype=np.float64)
    valid = ~np.isnan(c)
    if valid.sum() < 3:
        return float("nan"), float("nan"), float("nan")
    cc = c.copy()
    cc[~valid] = np.nanmin(c)
    half = len(cc) // 2
    peak_idx = int(np.nanargmax(cc))
    peak_val = cc[peak_idx]
    base = np.nanmin(cc)
    hm = base + 0.5 * (peak_val - base)
    # walk left/right from the peak until below half-max
    l = peak_idx
    while l > 0 and cc[l - 1] >= hm:
        l -= 1
    r = peak_idx
    while r < len(cc) - 1 and cc[r + 1] >= hm:
        r += 1
    fwhm = float(r - l + 1)
    offset = float(peak_idx - half)
    return fwhm, offset, float(peak_val)
