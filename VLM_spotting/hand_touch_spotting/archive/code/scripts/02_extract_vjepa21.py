"""Extract per-frame V-JEPA 2.1 ViT-B motion tokens (N,768) for HOI4D videos.

Dual-GPU: launch once per card with --num-shards 2 --shard-id {0,1}. Skips videos
already extracted. Video list = union of the video_ids in the given annotation
splits (default: val, for the Phase-0 diagnostic).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from data import vjepa_interleave as V  # noqa: E402

FRAMES_DIR = os.environ.get(
    "TOUCH_FRAMES_DIR",
    "/home/chang_noroot/data2/huyanh/Workspace/dataset/hoi4d/frames")


def video_ids_from_splits(splits):
    ann = os.path.join(ROOT, "data", "annotations")
    ids = []
    for sp in splits:
        for v in json.load(open(os.path.join(ann, f"{sp}.json"))):
            ids.append(v["video_id"])
    seen, uniq = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val"])
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs", "vjepa_feats"))
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch-windows", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ids = video_ids_from_splits(args.splits)
    ids = _shard(ids, args.shard_id, args.num_shards)
    if args.limit:
        ids = ids[: args.limit]

    enc = V.load_encoder(device="cuda", dtype=torch.float32)
    print(f"[extract] {len(ids)} videos | shard {args.shard_id+1}/{args.num_shards} "
          f"| out={args.out_dir}", flush=True)
    t0 = time.time()
    done = 0
    for i, vid in enumerate(ids):
        out_path = os.path.join(args.out_dir, vid + ".npy")
        if not args.overwrite and os.path.exists(out_path):
            continue
        clip_dir = os.path.join(FRAMES_DIR, vid)
        if not os.path.isdir(clip_dir):
            print(f"  MISSING frames: {vid}", flush=True)
            continue
        frames = _load_frames(clip_dir)
        feat = V.extract_video(frames, enc, device="cuda",
                               batch_windows=args.batch_windows, dtype=torch.float32)
        import numpy as np
        np.save(out_path, feat.astype(np.float16))
        done += 1
        if done % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(ids)}] {vid} {feat.shape} "
                  f"| {(time.time()-t0)/max(done,1):.2f}s/vid", flush=True)
    print(f"[extract] shard {args.shard_id} done {done} vids in "
          f"{(time.time()-t0)/60:.1f} min", flush=True)


def _shard(items, shard_id, num_shards):
    if num_shards <= 1:
        return items
    return [x for j, x in enumerate(items) if j % num_shards == shard_id]


def _load_frames(clip_dir):
    import glob
    import numpy as np
    from PIL import Image
    paths = sorted(glob.glob(os.path.join(clip_dir, "*.jpg")))
    return [np.asarray(Image.open(p).convert("RGB")) for p in paths]


if __name__ == "__main__":
    main()
