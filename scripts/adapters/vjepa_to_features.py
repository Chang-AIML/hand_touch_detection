"""Convert raw V-JEPA 2.1 even/odd per-frame features into the per-video, frame-aligned
`<video>.npy` format consumed by the touch downstream (FeatureDataset / MS-TCN / ASFormer).

Background
----------
V-JEPA uses tubelet_size=2, so we extract two complementary half-rate streams
(see ../../feature_extraction/extract_vjepa21.py):

    <clip>_even.npy  (Ne, D)   rows align to frames 0,2,4,...   Ne = ceil(N/2)
    <clip>_odd.npy   (No, D)   rows align to frames 1,3,5,...   No = floor(N/2)

The downstream head needs ONE frame-aligned array `<video>.npy` of shape [N, D]
(or [N, V, D] for the multi-version path). This adapter is the bridge, with modes:

    interleave : out[0::2]=even, out[1::2]=odd            -> [N, D]   (exact per-frame; default)
    even       : even stream only, time-interpolated x2   -> [N, D]   (odd frames invented)
    odd        : odd  stream only, time-interpolated x2   -> [N, D]   (even frames invented)
    stack      : both streams interp'd to N, as 2 versions -> [N, 2, D]
                 (FeatureDataset picks a version at random in training, version 0 at eval,
                  so the head can "decide" how to use even vs odd)

Targets and naming come from the label splits, so ONLY the labelled videos are written
and the output length always equals `num_frames` (perfect alignment for event['frame']).

Usage
-----
    python scripts/adapters/vjepa_to_features.py \
        --raw-dir   ../feature_extraction/VJEPA_feature \
        --out-dir   outputs/VJEPA_features \
        --label-dir data/HOI4D-v3 \
        --mode      interleave

Then train the head directly on it (no model code changes; feat dim auto-detected):
    python downstream/train_head.py -m mstcn --feat_dir /path/to/vjepa_features
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _interp_stream(feat: np.ndarray, src_idx: np.ndarray, n: int, kind: str) -> np.ndarray:
    """Resample `feat` (S, D) given each row's frame index `src_idx` (len S) onto frames
    0..n-1. Linear or nearest. Out-of-range targets clamp to the nearest endpoint."""
    tgt = np.arange(n)
    if kind == "nearest":
        pos = np.searchsorted(src_idx, tgt).clip(0, len(src_idx) - 1)
        left = (pos - 1).clip(0, len(src_idx) - 1)
        take_left = np.abs(tgt - src_idx[left]) <= np.abs(src_idx[pos] - tgt)
        pos = np.where(take_left, left, pos)
        return feat[pos]
    out = np.empty((n, feat.shape[1]), dtype=np.float32)
    for d in range(feat.shape[1]):                       # np.interp clamps ends automatically
        out[:, d] = np.interp(tgt, src_idx, feat[:, d])
    return out


def build_feature(even, odd, n: int, mode: str, interp: str) -> np.ndarray:
    """Return [N, D] (interleave/even/odd) or [N, 2, D] (stack)."""
    ne = (n + 1) // 2                                     # even-frame count: frames 0,2,...
    no = n // 2                                           # odd-frame  count: frames 1,3,...
    even_idx = np.arange(0, n, 2)                         # 0,2,4,...  (len ne)
    odd_idx = np.arange(1, n, 2)                          # 1,3,5,...  (len no)

    if mode == "interleave":
        if even is None or odd is None:
            raise ValueError("interleave needs both even and odd")
        D = even.shape[1]
        out = np.empty((n, D), dtype=np.float32)
        out[even_idx] = even[:ne]
        out[odd_idx] = odd[:no]
        return out

    if mode == "even":
        return _interp_stream(even[:ne], even_idx, n, interp)
    if mode == "odd":
        return _interp_stream(odd[:no], odd_idx, n, interp)

    if mode == "stack":                                  # [N, 2, D]: ver0=even, ver1=odd
        e = _interp_stream(even[:ne], even_idx, n, interp)
        o = _interp_stream(odd[:no], odd_idx, n, interp)
        return np.stack([e, o], axis=1)

    raise ValueError(f"unknown mode {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="dir with <clip>_even.npy / <clip>_odd.npy")
    ap.add_argument("--out-dir", required=True, help="output dir of per-video <video>.npy")
    ap.add_argument("--label-dir", required=True, help="holds {train,val,test}.json")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--mode", choices=["interleave", "even", "odd", "stack"], default="interleave")
    ap.add_argument("--interp", choices=["linear", "nearest"], default="linear",
                    help="time interpolation for even/odd/stack modes")
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    videos = {}                                          # video_id -> num_frames
    for sp in args.splits:
        for e in json.load(open(os.path.join(args.label_dir, f"{sp}.json"))):
            videos[e["video"]] = e["num_frames"]
    print(f"[vjepa-adapter] mode={args.mode} interp={args.interp} | {len(videos)} videos "
          f"| raw={args.raw_dir} -> out={args.out_dir}")

    need_even = args.mode in ("interleave", "even", "stack")
    need_odd = args.mode in ("interleave", "odd", "stack")
    done = miss = 0
    for vid, n in sorted(videos.items()):
        out_path = os.path.join(args.out_dir, vid.replace("/", "__") + ".npy")
        if not args.overwrite and os.path.exists(out_path):
            done += 1
            continue
        ev_p = os.path.join(args.raw_dir, f"{vid}_even.npy")
        od_p = os.path.join(args.raw_dir, f"{vid}_odd.npy")
        if (need_even and not os.path.exists(ev_p)) or (need_odd and not os.path.exists(od_p)):
            miss += 1
            continue
        even = np.load(ev_p).astype(np.float32) if need_even else None
        odd = np.load(od_p).astype(np.float32) if need_odd else None
        feat = build_feature(even, odd, n, args.mode, args.interp)
        np.save(out_path, feat.astype(args.dtype))
        done += 1

    print(f"[vjepa-adapter] wrote/kept {done} | missing-raw {miss} | shape example "
          f"[N,{'2,' if args.mode=='stack' else ''}D]")


if __name__ == "__main__":
    main()
