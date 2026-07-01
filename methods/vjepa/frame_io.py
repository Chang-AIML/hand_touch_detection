"""Shared helpers for per-frame feature extraction on the HOI4D frame folders.

Each "clip" is a directory of zero-padded JPGs (000000.jpg, 000001.jpg, ...).
We produce ONE feature per frame, so the output array's first axis length always
equals the number of JPGs in the folder.
"""
from __future__ import annotations

import os
import glob
import numpy as np
from PIL import Image


# ----------------------------------------------------------------------------- clip / frame discovery
def list_clips(frames_root: str):
    """Return sorted (clip_id, clip_dir) for every sub-folder that holds frames."""
    clips = []
    for name in sorted(os.listdir(frames_root)):
        d = os.path.join(frames_root, name)
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*.jpg")):
            clips.append((name, d))
    return clips


def list_frame_paths(clip_dir: str):
    return sorted(glob.glob(os.path.join(clip_dir, "*.jpg")))


def load_frames_rgb(clip_dir: str):
    """Load all frames of a clip as a list of HxWx3 uint8 numpy arrays (RGB)."""
    return [np.asarray(Image.open(p).convert("RGB")) for p in list_frame_paths(clip_dir)]


def shard(items, shard_id: int, num_shards: int):
    """Split a list into `num_shards` contiguous shards, return shard `shard_id`."""
    if num_shards <= 1:
        return items
    bounds = np.linspace(0, len(items), num_shards + 1).astype(int)
    return items[bounds[shard_id]: bounds[shard_id + 1]]


# ----------------------------------------------------------------------------- pooling helper (V-JEPA)
def pool_features(feat: np.ndarray, mode: str = "temporal") -> np.ndarray:
    """Pool a V-JEPA per-frame feature array.

    Accepts the spatial-temporal feature saved by extract_vjepa.py, shaped either
        (N, Hp, Wp, D)   -- spatial grid kept (Hp=Wp=16 for vitl/256), or
        (N, Hp*Wp, D)    -- flattened spatial tokens.

    mode:
        "spatiotemporal" -> return the full grid unchanged (N, Hp, Wp, D).
        "temporal"       -> mean-pool over the spatial axes -> (N, D),
                            i.e. one vector per frame (drop-in for a sequence head).
    """
    if feat.ndim == 3:                       # (N, Hp*Wp, D)
        if mode == "temporal":
            return feat.mean(axis=1)
        return feat
    if feat.ndim == 4:                       # (N, Hp, Wp, D)
        if mode == "temporal":
            return feat.mean(axis=(1, 2))
        return feat
    raise ValueError(f"pool_features expects a (N,Hp,Wp,D) or (N,P,D) array, got shape {feat.shape}")


# ----------------------------------------------------------------------------- io
def save_feature(out_dir: str, clip_id: str, feat: np.ndarray, dtype=np.float16):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, clip_id + ".npy"), feat.astype(dtype))


def already_done(out_dir: str, clip_id: str) -> bool:
    return os.path.exists(os.path.join(out_dir, clip_id + ".npy"))
