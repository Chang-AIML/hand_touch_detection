"""PROBE data: extract UNPOOLED token grids (576,768) for frames near GT events
(offset -R..+R) on a subset of train videos. Saves one npz for the recoverability
probe: does a query-conditioned attention-pool separate the exact contact frame from
its neighbors, where mean-pool cannot?"""
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
from data import vjepa_interleave as V                            # noqa: E402
from data.vjepa_grid import extract_video_grid                    # noqa: E402
from PIL import Image                                             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_videos", type=int, default=90)
    ap.add_argument("--R", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs/probe/grids.npz"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    vids = json.load(open(os.path.join(LAB, "train.json")))
    vids = [v for v in vids if os.path.isdir(os.path.join(FRAMES, v["video"]))
            and len(v["events"]) >= 3]
    vids = vids[:args.n_videos]
    enc = V.load_encoder(device="cuda", dtype=torch.float32)

    X, off, typ, vidix = [], [], [], []
    types = {"touch": 0, "untouch": 1}
    import time
    t0 = time.time()
    for vi, v in enumerate(vids):
        fr = sorted(glob.glob(os.path.join(FRAMES, v["video"], "*.jpg")))
        frames = [np.asarray(Image.open(p).convert("RGB")) for p in fr]
        N = len(frames)
        grid = extract_video_grid(frames, enc, device="cuda", batch_windows=8, dtype=torch.float32)
        for e in v["events"]:
            g = e["frame"]
            for o in range(-args.R, args.R + 1):
                f = g + o
                if 0 <= f < N and f < grid.shape[0]:
                    X.append(grid[f].astype(np.float16)); off.append(o)
                    typ.append(types[e["label"]]); vidix.append(vi)
        if (vi + 1) % 10 == 0:
            print(f"  [{vi+1}/{len(vids)}] samples={len(X)} | {(time.time()-t0)/(vi+1):.2f}s/vid", flush=True)
    X = np.stack(X); off = np.array(off, np.int8); typ = np.array(typ, np.int8); vidix = np.array(vidix, np.int16)
    np.savez(args.out, X=X, off=off, typ=typ, vidix=vidix)
    print(f"[saved] {args.out}  X={X.shape} ({X.nbytes/1e9:.2f}GB) | "
          f"videos={len(vids)} events-frames={len(off)}", flush=True)


if __name__ == "__main__":
    main()
