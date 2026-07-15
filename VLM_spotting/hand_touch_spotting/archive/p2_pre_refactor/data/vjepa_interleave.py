"""CORE §3.4 — V-JEPA 2.1 per-frame motion tokens via even/odd interleave.

V-JEPA 2.1's patch embedding is a Conv3d with time kernel = stride = TUBELET = 2,
so a T-frame clip yields only T/2 temporal tokens (each token covers 2 adjacent
frames). To recover a *real* feature for every frame (no interpolation) we run the
frozen encoder TWICE with a 1-frame temporal offset and interleave:

    even pass: frames [0,1,2,3,...] -> tubelet (0,1)(2,3)... -> assign to frames 0,2,4,...
    odd  pass: frames [1,2,3,4,...] -> tubelet (1,2)(3,4)... -> assign to frames 1,3,5,...
    interleave: out[2k] = even[k];  out[2k+1] = odd[k]

Property preserved (see exp_plan §3.4): feature of ANY frame t = tubelet (t, t+1) =
the forward 2-frame motion feature of "this frame + its next". Semantics uniform
across all frames; every frame is a genuinely-forwarded real feature.

This module is import-safe with NO heavy deps at module load; torch/model imports
happen inside functions so the interleave unit test runs standalone.
"""
from __future__ import annotations

import os
import sys
from typing import List

import numpy as np

# ---- V-JEPA 2.1 ViT-B constants (native repo, vjepa2_1_vit_base_384) -----------
VJEPA_REPO = os.environ.get(
    "VJEPA_REPO", "/home/chang_noroot/data2/huyanh/Workspace/repos/vjepa2")
VJEPA_CKPT = os.environ.get(
    "VJEPA_CKPT",
    "/home/chang_noroot/data2/huyanh/Workspace/repos/feature_extraction/ckpts/"
    "vjepa2_1_vitb_dist_vitG_384.pt")
IMG_SIZE = 384
PATCH = 16
TUBELET = 2
GRID = IMG_SIZE // PATCH          # 24 spatial tokens per side
DIM = 768                         # ViT-B embed dim
WIN = 16                          # frames per encoder window (multiple of TUBELET)
TOK_T = WIN // TUBELET            # temporal tokens per window
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================ pure interleave (unit-tested)
def interleave(even: np.ndarray, odd: np.ndarray, n_frames: int) -> np.ndarray:
    """Assemble per-frame features from the two half-rate passes.

    even[k] -> frame 2k, odd[k] -> frame 2k+1. Returns exactly (n_frames, ...).
    Ordering is the §3.4 invariant: out[0::2]=even, out[1::2]=odd.
    """
    assert even.shape[1:] == odd.shape[1:], (even.shape, odd.shape)
    out = np.empty((n_frames,) + even.shape[1:], dtype=even.dtype)
    ev = np.arange(0, n_frames, 2)
    od = np.arange(1, n_frames, 2)
    out[ev] = even[: ev.shape[0]]
    out[od] = odd[: od.shape[0]]
    return out


# ============================================================ model / preprocessing
def load_encoder(device="cuda", dtype=None):
    """Load the frozen V-JEPA 2.1 ViT-B encoder from the local checkpoint.

    Runs in fp32 by default (RoPE dtype quirk in the native repo). Returns an
    eval-mode, requires_grad=False encoder.
    """
    import torch
    if VJEPA_REPO not in sys.path:
        sys.path.insert(0, VJEPA_REPO)
    from src.hub.backbones import vjepa2_1_vit_base_384, _clean_backbone_key

    encoder, _predictor = vjepa2_1_vit_base_384(pretrained=False)
    sd = torch.load(VJEPA_CKPT, map_location="cpu", weights_only=False)
    enc_sd = _clean_backbone_key(sd["ema_encoder"])
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    # pos_embed keys may be absent (RoPE) — tolerate, but nothing else should miss.
    hard_missing = [k for k in missing if "pos_embed" not in k]
    assert not hard_missing, f"unexpected missing keys: {hard_missing[:8]}"
    encoder = encoder.to(device)
    if dtype is not None:
        encoder = encoder.to(dtype)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def preprocess_frames(frames_rgb: List[np.ndarray]):
    """List of HxWx3 uint8 RGB -> tensor (N, 3, 384, 384), ImageNet-normalised.

    Shortest-edge resize to 384 then center-crop 384 (square egocentric frames are
    already close to square; this keeps the hand-object region centred).
    """
    import torch
    import torch.nn.functional as F
    arr = np.stack(frames_rgb).astype(np.float32) / 255.0        # (N,H,W,3)
    t = torch.from_numpy(arr).permute(0, 3, 1, 2)                # (N,3,H,W)
    h, w = t.shape[-2:]
    scale = IMG_SIZE / min(h, w)
    nh, nw = round(h * scale), round(w * scale)
    t = F.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False)
    top = (nh - IMG_SIZE) // 2
    left = (nw - IMG_SIZE) // 2
    t = t[:, :, top:top + IMG_SIZE, left:left + IMG_SIZE]        # (N,3,384,384)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (t - mean) / std


def _pad_to_multiple(x, m: int):
    """Pad axis-0 up to a multiple of m by repeating the last frame."""
    import torch
    n = x.shape[0]
    pad = (-n) % m
    if pad:
        x = torch.cat([x, x[-1:].expand(pad, *x.shape[1:])], dim=0)
    return x


def _run_pass(proc, encoder, device, batch_windows: int, dtype) -> np.ndarray:
    """proc: (M,3,384,384) preprocessed. Returns (T, DIM) spatial-mean-pooled
    per-temporal-token features; token p covers local frames (2p, 2p+1)."""
    import torch
    proc = _pad_to_multiple(proc, WIN)
    nwin = proc.shape[0] // WIN
    wins = proc.view(nwin, WIN, 3, IMG_SIZE, IMG_SIZE)
    chunks = []
    for s in range(0, nwin, batch_windows):
        b = wins[s:s + batch_windows].to(device, dtype, non_blocking=True)
        b = b.permute(0, 2, 1, 3, 4).contiguous()                # (bw,3,WIN,384,384)
        with torch.no_grad():
            h = encoder(b)                                       # (bw, TOK_T*GRID*GRID, DIM)
        bw = h.shape[0]
        h = h.view(bw, TOK_T, GRID * GRID, DIM).mean(dim=2)      # spatial mean -> (bw,TOK_T,DIM)
        chunks.append(h.float().cpu())
    h = torch.cat(chunks, dim=0)                                 # (nwin, TOK_T, DIM)
    return h.reshape(nwin * TOK_T, DIM).numpy()


def extract_video(frames_rgb, encoder, device="cuda", batch_windows: int = 8,
                  dtype=None) -> np.ndarray:
    """Frames -> per-frame V-JEPA 2.1 motion tokens (N_frames, DIM), float32.

    One real token per frame; token t = tubelet(t, t+1). No interpolation.
    """
    import torch
    if dtype is None:
        dtype = next(encoder.parameters()).dtype
    proc = preprocess_frames(frames_rgb)                         # (N,3,384,384)
    n = proc.shape[0]
    even = _run_pass(proc, encoder, device, batch_windows, dtype)      # frames 0,2,4,...
    odd = _run_pass(proc[1:], encoder, device, batch_windows, dtype)   # frames 1,3,5,...
    return interleave(even, odd, n)


# ============================================================ standalone unit test
def _selftest_interleave():
    """len==N_frames + correct even/odd placement (no model needed)."""
    for n in (1, 2, 7, 8, 15, 300):
        ne = (n + 1) // 2
        no = n // 2
        even = np.arange(ne, dtype=np.float32)[:, None] * 10        # marker 0,10,20..
        odd = np.arange(no, dtype=np.float32)[:, None] * 10 + 1     # marker 1,11,21..
        out = interleave(even, odd, n)
        assert out.shape[0] == n, (n, out.shape)
        for t in range(n):
            exp = (t // 2) * 10 + (t % 2)          # even frame->k*10, odd frame->k*10+1
            assert out[t, 0] == exp, (n, t, out[t, 0], exp)
    print("[selftest] interleave len==N and even/odd placement OK")


if __name__ == "__main__":
    _selftest_interleave()
