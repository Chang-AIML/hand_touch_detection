"""V-JEPA 2.1 per-frame UNPOOLED spatial-token grid (N_frames, 576, 768) via the
same even/odd interleave as vjepa_interleave, but WITHOUT the spatial mean-pool
(the `.mean(dim=2)` at vjepa_interleave.py:135 that smears cos 0.93 -> 0.995).
grid.mean(axis=1) must reproduce the cached feat_interleave (sanity-checked)."""
from __future__ import annotations

import numpy as np
import torch

from data import vjepa_interleave as V


def _run_pass_grid(proc, encoder, device, batch_windows, dtype):
    """proc:(M,3,384,384) -> (T, 576, 768) per-temporal-token spatial grid (NO pool)."""
    proc = V._pad_to_multiple(proc, V.WIN)
    nwin = proc.shape[0] // V.WIN
    wins = proc.view(nwin, V.WIN, 3, V.IMG_SIZE, V.IMG_SIZE)
    chunks = []
    for s in range(0, nwin, batch_windows):
        b = wins[s:s + batch_windows].to(device, dtype, non_blocking=True)
        b = b.permute(0, 2, 1, 3, 4).contiguous()
        with torch.no_grad():
            h = encoder(b)                                              # (bw, TOK_T*576, 768)
        bw = h.shape[0]
        h = h.view(bw, V.TOK_T, V.GRID * V.GRID, V.DIM)                 # (bw,TOK_T,576,768) NO pool
        chunks.append(h.float().cpu())
    h = torch.cat(chunks, dim=0)                                        # (nwin,TOK_T,576,768)
    return h.reshape(nwin * V.TOK_T, V.GRID * V.GRID, V.DIM).numpy()


def extract_video_grid(frames_rgb, encoder, device="cuda", batch_windows=8, dtype=None):
    """Frames -> (N_frames, 576, 768) float32 unpooled grid. token t = tubelet(t,t+1)."""
    if dtype is None:
        dtype = next(encoder.parameters()).dtype
    proc = V.preprocess_frames(frames_rgb)
    n = proc.shape[0]
    even = _run_pass_grid(proc, encoder, device, batch_windows, dtype)
    odd = _run_pass_grid(proc[1:], encoder, device, batch_windows, dtype)
    return V.interleave(even, odd, n)                                   # (N,576,768)
