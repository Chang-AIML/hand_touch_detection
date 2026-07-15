"""Precompute FROZEN V-JEPA 2.1 ViT-B UNPOOLED grids for ALL videos, EVEN pass only
(1 encoder pass, ~N/2 tubelet tokens; interpolate to per-frame at load time). Stored
fp16 -> ~132MB/video, ~377GB for all 2850. Sharded for 2-GPU parallel. Feeds Stage-1
token-compression training (V-JEPA never re-run during training)."""
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
_COMMON = "/home/chang_noroot/data2/huyanh/Workspace/hand_touch_detection"
LAB = os.path.join(_COMMON, "data", "HOI4D-v3")
FRAMES = "/home/chang_noroot/data2/huyanh/Workspace/dataset/hoi4d/frames"
from concurrent.futures import ThreadPoolExecutor                  # noqa: E402
from data import vjepa_interleave as V                            # noqa: E402
from data.vjepa_grid import _run_pass_grid                         # noqa: E402
from PIL import Image                                             # noqa: E402


def _read_rgb(p):
    return np.asarray(Image.open(p).convert("RGB"))

OUT = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/grid_even"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--batch-windows", type=int, default=32)
    ap.add_argument("--io-workers", type=int, default=6)
    ap.add_argument("--cpu-threads", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(a.cpu_threads)                          # cap to stay under core count (2 procs)

    vids = []
    for sp in a.splits:
        for v in json.load(open(os.path.join(LAB, f"{sp}.json"))):
            vid = v["video"]
            if os.path.isdir(os.path.join(FRAMES, vid)):
                vids.append((vid, v.get("num_frames", 0)))
    seen = set(); uniq = []
    for vid, nf in vids:
        if vid not in seen:
            seen.add(vid); uniq.append((vid, nf))
    vids = [x for j, x in enumerate(uniq) if j % a.nshard == a.shard]

    enc = V.load_encoder(device="cuda", dtype=torch.float32)
    print(f"[even-grid] shard {a.shard+1}/{a.nshard} | {len(vids)} videos | out={OUT}", flush=True)
    t0 = time.time(); done = 0
    pool = ThreadPoolExecutor(max_workers=a.io_workers)
    for i, (vid, _) in enumerate(vids):
        out_path = os.path.join(OUT, vid.replace("/", "__") + ".npy")
        if os.path.exists(out_path):
            continue
        fr = sorted(glob.glob(os.path.join(FRAMES, vid, "*.jpg")))
        frames = list(pool.map(_read_rgb, fr))                     # parallel jpg decode (I/O bound)
        proc = V.preprocess_frames(frames)                         # (N,3,384,384)
        # EVEN pass only: 1 encoder pass -> (~N/2, 576, 768) unpooled (token k covers frames 2k,2k+1)
        even = _run_pass_grid(proc, enc, "cuda", a.batch_windows, torch.float32)
        np.save(out_path, even.astype(np.float16))
        done += 1
        if done % 25 == 0:
            el = time.time() - t0
            print(f"  [{i+1}/{len(vids)}] {vid} {even.shape} | {el/done:.2f}s/vid "
                  f"| ETA {(len(vids)-i-1)*el/done/60:.0f}min", flush=True)
    print(f"[even-grid] shard {a.shard} done {done} in {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
