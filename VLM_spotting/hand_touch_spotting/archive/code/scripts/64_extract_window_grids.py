"""Extract UNPOOLED token grids (576,768) only for the union of ±E frames around each
stage-1 idx and each GT event, per video (compact). Feeds the end-to-end attention-pool
refine test. Saves per-video npz {frames:(n,), grids:(n,576,768) fp16} to a scratch dir."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/home/chang_noroot/data2/hf_cache")
_COMMON = "/home/chang_noroot/data2/huyanh/Workspace/hand_touch_detection"
LAB = os.path.join(_COMMON, "data", "HOI4D-v3")
FRAMES = "/home/chang_noroot/data2/huyanh/Workspace/dataset/hoi4d/frames"
CACHE = os.path.join(ROOT, "outputs/action/cache")
from data import vjepa_interleave as V                            # noqa: E402
from data.vjepa_grid import extract_video_grid                    # noqa: E402
from PIL import Image                                             # noqa: E402

OUT = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/grid_win"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--E", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cache_name = "idx_multi_train_800" if a.split == "train" else f"idx_multi_{a.split}_0"
    coarse = torch.load(os.path.join(CACHE, cache_name + ".pt"), weights_only=False)["coarse"]
    lab = "train" if a.split == "train" else a.split
    vids = json.load(open(os.path.join(LAB, f"{lab}.json")))
    vids = [v for v in vids if os.path.isdir(os.path.join(FRAMES, v["video"]))]
    if a.limit:
        vids = vids[:a.limit]
    vids = [v for j, v in enumerate(vids) if j % a.nshard == a.shard]

    enc = V.load_encoder(device="cuda", dtype=torch.float32)
    import time
    t0 = time.time(); done = 0
    for vi, v in enumerate(vids):
        k = v["video"].replace("/", "__")
        out_path = os.path.join(OUT, f"{k}.npz")
        if os.path.exists(out_path):
            continue
        N = v["num_frames"]
        need = set()
        for t in ("touch", "untouch"):
            centers = [int(p) for p, _ in coarse.get((k, t), [])] + \
                      [e["frame"] for e in v["events"] if e["label"] == t]
            for c in centers:
                need.update(range(max(0, c - a.E), min(N, c + a.E + 1)))
        if not need:
            continue
        fr = sorted(glob.glob(os.path.join(FRAMES, v["video"], "*.jpg")))
        frames = [np.asarray(Image.open(p).convert("RGB")) for p in fr]
        grid = extract_video_grid(frames, enc, device="cuda", batch_windows=8, dtype=torch.float32)
        idxs = sorted(f for f in need if f < grid.shape[0])
        np.savez(out_path, frames=np.array(idxs, np.int32),
                 grids=grid[idxs].astype(np.float16))
        done += 1
        if done % 20 == 0:
            print(f"  [{vi+1}/{len(vids)}] {k} nframes={len(idxs)} | "
                  f"{(time.time()-t0)/done:.2f}s/vid", flush=True)
    print(f"[done] split={a.split} shard={a.shard} extracted {done} videos "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
