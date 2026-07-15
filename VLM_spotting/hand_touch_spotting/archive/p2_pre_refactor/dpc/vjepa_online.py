"""ON-THE-FLY FROZEN V-JEPA 2.1 ViT-B even-pass grid extractor (no disk).

This is the *online* twin of the offline extractor `scripts/67_extract_even_grids.py`.
It produces the SAME "even-pass grid" tokens the offline script writes to
`vjepa/grid_even/<vid>.npy`, but computed inside a forward pass from RGB frames, so
downstream (FrameCompress / IdxLocalizer) weights trained on the cached grids stay
valid.

Exact-reproduction contract (why this is bit-faithful to the offline cache)
--------------------------------------------------------------------------
Rather than re-implement the grid math, this module REUSES the exact ground-truth
functions the offline pipeline calls:

  * preprocessing  -> `data.vjepa_interleave.preprocess_frames`  (interleave.py:89-108)
                      384 shortest-edge resize + center-crop, ImageNet mean/std.
  * encoder load   -> `data.vjepa_interleave.load_encoder`        (interleave.py:62-86)
                      repo id 'vjepa2_1_vit_base_384', ema_encoder state-dict,
                      requires_grad_(False), eval(), fp32 (RoPE dtype quirk).
  * grid pass      -> `data.vjepa_grid._run_pass_grid`            (vjepa_grid.py:13-28)
                      pad-to-WIN=16, tubelet=2, 24x24=576 tokens, 768-dim, NO pool.

The offline script (67_extract_even_grids.py:59,69-72) does, in fp32:
    enc  = V.load_encoder(device="cuda", dtype=torch.float32)
    proc = V.preprocess_frames(frames)                    # (N,3,384,384)
    even = _run_pass_grid(proc, enc, "cuda", bw, torch.float32)   # EVEN pass only
    np.save(out, even.astype(np.float16))                 # (~N/2, 576, 768) fp16

`extract_grid` runs that identical even-pass computation and returns exactly the
first ceil(T/2) tubelet tokens (token k = tubelet(2k, 2k+1)), i.e. the per-even-frame
grid. Interpolation to N frames is intentionally NOT done here — that is the
downstream FrameCompress' job (frame_compress.py:_interp), matching the cached path
where the dataset hands the raw (N/2, 576, 768) even grid straight to the compressor.

dtype note
----------
The ENCODER is always run in fp32 (a) to bit-match the offline cache and (b) because
the native V-JEPA repo requires fp32 for its RoPE (see load_encoder docstring). The
`dtype` ctor arg is the RETURN dtype of the grid tokens only (default bf16 for the
online LLM stack; pass torch.float16 to match the on-disk fp16 grid exactly). An
outer autocast context is explicitly disabled around the encoder so the compute stays
fp32 regardless of caller context (the offline extractor used no autocast).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn

# ---- staged weights (this machine) ------------------------------------------------
# Defaults point at the locally staged V-JEPA repo + checkpoint. Set via env exactly
# like the existing code (data/vjepa_interleave.py:28-33 reads VJEPA_REPO/VJEPA_CKPT
# at import time), so we must publish these BEFORE importing vjepa_interleave.
DEFAULT_REPO = "/home/chang/Project/vlm_deps/vjepa2"
DEFAULT_CKPT = "/home/chang/Project/vlm_deps/vjepa2_1_vitb_dist_vitG_384.pt"
os.environ.setdefault("VJEPA_REPO", DEFAULT_REPO)
os.environ.setdefault("VJEPA_CKPT", DEFAULT_CKPT)

# ---- make the HTS repo root importable (mirrors scripts/67_extract_even_grids.py) --
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                       # .../hand_touch_spotting
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data import vjepa_interleave as V               # noqa: E402
from data.vjepa_grid import _run_pass_grid           # noqa: E402  (ground-truth even-pass grid)


class OnlineVJEPA(nn.Module):
    """Frozen V-JEPA 2.1 ViT-B even-pass grid extractor, computed on-the-fly.

    Parameters
    ----------
    device : str | torch.device
        Where the encoder lives and where the grid is returned.
    dtype : torch.dtype
        RETURN dtype of the grid tokens (default bf16). The encoder itself always
        runs in fp32 to reproduce the offline cache bit-for-bit and to satisfy the
        native RoPE fp32 requirement. Pass torch.float16 to match the on-disk grid.
    repo, ckpt : str | None
        V-JEPA repo dir / checkpoint path. None -> env VJEPA_REPO / VJEPA_CKPT
        (which default to the staged paths above).
    """

    def __init__(self, device, dtype: torch.dtype = torch.bfloat16, repo=None, ckpt=None,
                 compute_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.device = torch.device(device)
        self.out_dtype = dtype
        # fp32 = bit-match the offline cache. bf16 = ~2x faster: run under autocast so
        # matmuls use bf16 while RoPE/norm stay fp32 (small drift, acceptable for fine-tuning).
        self.compute_dtype = compute_dtype

        repo = repo or os.environ.get("VJEPA_REPO", DEFAULT_REPO)
        ckpt = ckpt or os.environ.get("VJEPA_CKPT", DEFAULT_CKPT)
        # load_encoder resolves these as module-level globals at CALL time, so
        # overriding them here lets a caller point at a different repo/ckpt without
        # touching the environment. (data/vjepa_interleave.py:69-74)
        V.VJEPA_REPO = repo
        V.VJEPA_CKPT = ckpt

        # fp32 encoder == exactly what scripts/67_extract_even_grids.py loads.
        encoder = V.load_encoder(device=self.device, dtype=torch.float32)
        encoder.requires_grad_(False)
        encoder.eval()
        self.encoder = encoder

    # keep the frozen encoder in eval() even if the parent module is put in train()
    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    @staticmethod
    def _to_frame_list(frames):
        """(T,H,W,3) np.uint8 | torch.Tensor -> list[HxWx3 uint8], matching the
        list-of-decoded-jpgs that preprocess_frames expects (interleave.py:97)."""
        if isinstance(frames, torch.Tensor):
            frames = frames.detach().cpu().numpy()
        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"expected frames (T,H,W,3), got {frames.shape}")
        if frames.dtype != np.uint8:
            frames = frames.astype(np.uint8)
        return [frames[i] for i in range(frames.shape[0])]

    @torch.no_grad()
    def extract_grid(self, frames_uint8, batch_windows: int = 32) -> torch.Tensor:
        """RGB window -> even-pass grid (ceil(T/2), 576, 768) on self.device.

        Reproduces the offline even grid: token k covers tubelet(2k, 2k+1). No
        interpolation here (downstream FrameCompress interpolates ceil(T/2) -> T).

        For even T (e.g. the smoke test T=4) the length is exactly T//2.
        """
        self.encoder.eval()
        frames = self._to_frame_list(frames_uint8)
        T = len(frames)

        # compute_dtype fp32 -> disable autocast, bit-match offline cache (67_extract_even_grids.py).
        # compute_dtype bf16 -> enable autocast(bf16): matmuls run bf16 (~2x faster) while
        # autocast keeps RoPE / layernorm in fp32, so the native RoPE fp32 requirement holds.
        _amp = self.compute_dtype != torch.float32
        with torch.autocast(device_type=self.device.type, enabled=_amp, dtype=self.compute_dtype):
            proc = V.preprocess_frames(frames)                       # (T,3,384,384) fp32
            # EVEN pass only -> identical call shape to 67_extract_even_grids.py:71.
            even = _run_pass_grid(proc, self.encoder, self.device,
                                  batch_windows, torch.float32)       # (nwin*TOK_T,576,768) np

        n_even = (T + 1) // 2                                         # #even-indexed frames
        even = even[:n_even]                                         # drop pad-tail tokens
        return torch.from_numpy(np.ascontiguousarray(even)).to(self.device, self.out_dtype)


# ==================================================================== smoke test (CPU ok)
if __name__ == "__main__":
    dev = "cpu"
    print(f"[smoke] loading OnlineVJEPA on {dev} (fp32 encoder) "
          f"repo={os.environ['VJEPA_REPO']} ckpt={os.environ['VJEPA_CKPT']}", flush=True)
    model = OnlineVJEPA(device=dev, dtype=torch.bfloat16)

    x = np.random.RandomState(0).randint(0, 256, size=(4, 384, 384, 3), dtype=np.uint8)
    grid = model.extract_grid(x)

    print(f"[smoke] out shape={tuple(grid.shape)} dtype={grid.dtype} device={grid.device}")
    print(f"[smoke] checksum(fp32 sum)={grid.float().sum().item():.4f} "
          f"mean={grid.float().mean().item():.6f} std={grid.float().std().item():.6f}")
    assert tuple(grid.shape) == (2, 576, 768), grid.shape
    print("[smoke] PASS: shape == (2, 576, 768)")
