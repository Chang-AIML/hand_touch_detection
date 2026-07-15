"""Extract FROZEN V-JEPA 2.1 ViT-B unpooled EVEN-pass grids for the TACO videos in the
touchmoment annotation (the C*-prefixed ids; HOI4D H* already have features). Same format
as scripts/67 -> written into the SAME grid_even dir, keyed by video id (no collision).
Enables the '8-query compression generality' probe: does the HOI4D-trained connector still
produce contact-decodable tokens on unseen TACO? Single-GPU friendly (GPU0 is another user)."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/home/chang_noroot/data2/hf_cache")
TM = "/home/chang_noroot/data2/huyanh/Workspace/dataset/hoi4d/touchmoment"
FRAMES = "/home/chang_noroot/data2/huyanh/Workspace/dataset/hoi4d/frames"
OUT = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/grid_even"
from concurrent.futures import ThreadPoolExecutor                  # noqa: E402
from data import vjepa_interleave as V                            # noqa: E402
from data.vjepa_grid import _run_pass_grid                         # noqa: E402
from PIL import Image                                             # noqa: E402


def _read_rgb(p):
    return np.asarray(Image.open(p).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--batch-windows", type=int, default=32)
    ap.add_argument("--io-workers", type=int, default=6); ap.add_argument("--cpu-threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)               # 0 = all TACO
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); torch.set_num_threads(a.cpu_threads)

    ids = {}
    for sp in ("train", "val", "test"):
        for v in json.load(open(os.path.join(TM, f"{sp}.json"))):
            ids.setdefault(v["video"], v.get("num_frames", 0))
    taco = [vid for vid in ids if not vid.startswith("H")]        # C* = TACO
    taco = [vid for vid in taco if os.path.isdir(os.path.join(FRAMES, vid))
            and not os.path.exists(os.path.join(OUT, vid.replace("/", "__") + ".npy"))]
    taco = sorted(taco)
    if a.limit:
        taco = taco[:a.limit]
    vids = [vid for j, vid in enumerate(taco) if j % a.nshard == a.shard]

    enc = V.load_encoder(device="cuda", dtype=torch.float32)
    print(f"[taco-grid] shard {a.shard+1}/{a.nshard} | {len(vids)} TACO videos to extract | out={OUT}", flush=True)
    t0 = time.time(); done = 0
    pool = ThreadPoolExecutor(max_workers=a.io_workers)
    for i, vid in enumerate(vids):
        out_path = os.path.join(OUT, vid.replace("/", "__") + ".npy")
        if os.path.exists(out_path):
            continue
        fr = sorted(glob.glob(os.path.join(FRAMES, vid, "*.jpg")))
        if not fr:
            print(f"  [skip-noframes] {vid}", flush=True); continue
        frames = list(pool.map(_read_rgb, fr))
        proc = V.preprocess_frames(frames)
        even = _run_pass_grid(proc, enc, "cuda", a.batch_windows, torch.float32)
        np.save(out_path, even.astype(np.float16))
        done += 1
        if done % 20 == 0:
            el = time.time() - t0
            print(f"  [{i+1}/{len(vids)}] {vid} {even.shape} | {el/done:.2f}s/vid "
                  f"| ETA {(len(vids)-i-1)*el/done/60:.0f}min", flush=True)
    print(f"[taco-grid] shard {a.shard} DONE {done} in {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
