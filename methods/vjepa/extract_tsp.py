"""Per-frame feature extraction with TSP (R(2+1)D-34), HOI4D-finetuned checkpoint.

TSP's R(2+1)D backbone global-pools space AND time, emitting ONE 512-d vector per
clip -- there is no tubelet to interleave. Following the E2E-Spot recipe ("we
extract per-frame features by densely striding a window around each frame",
supplementary B.2/B.3), we get per-frame features with a dense stride-1 sliding
window centered on each frame:

    frame t  ->  window [t-8, ..., t+7]  (16 frames, indices clamped to [0, N-1])
             ->  R(2+1)D-34  ->  512-d feature assigned to frame t

So the output is exactly (N, 512), one feature per JPG. WIN matches the
checkpoint's training clip length (CLIP_LEN=16 in train_tsp_on_hoi4d.sh).

We load the finetuned backbone directly from the checkpoint (no Kinetics download:
the checkpoint already holds every backbone weight).

Run with the `dev` conda env:
    /data/dong/miniconda3/envs/dev/bin/python extract_tsp.py \
        --frames-dir /data/dong/project/Workspace/dataset/hoi4d/frames \
        --out-dir    ./output/tsp
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torchvision
from torch import nn

import frame_io

# default locations
TSP_REPO = "/data/dong/project/Workspace/repos/TSP"
DEFAULT_CKPT = "/data/dong/project/repos/TSP/train/output/r2plus1d_34-tsp_on_hoi4d/checkpoint.pth"
WIN = 16                       # clip length used to train the HOI4D checkpoint


def build_backbone(ckpt_path: str, device):
    """Build R(2+1)D-34 and load the finetuned backbone weights (offline)."""
    sys.path.insert(0, TSP_REPO)
    from models.backbone import r2plus1d_34

    model = r2plus1d_34(pretrained=False)   # no Kinetics download; we overwrite weights anyway
    model.fc = nn.Sequential()              # -> forward returns the 512-d pooled feature

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    # TSP stores the backbone under "features." inside its Model wrapper; strip that prefix
    # and drop the classifier head ("fc.*").
    bb = {k[len("features."):]: v for k, v in state.items() if k.startswith("features.")}
    if not bb:  # checkpoint already a bare backbone
        bb = {k: v for k, v in state.items() if not k.startswith("fc.")}
    missing, unexpected = model.load_state_dict(bb, strict=False)
    missing = [m for m in missing if not m.startswith("fc.")]
    assert not missing and not unexpected, f"weight mismatch: missing={missing} unexpected={unexpected}"
    return model.to(device).eval()


def build_transform():
    """Match TSP's eval pipeline (extract_features.py)."""
    sys.path.insert(0, TSP_REPO)
    from common import transforms as T
    normalize = T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                            std=[0.22803, 0.22145, 0.216989])
    return torchvision.transforms.Compose([
        T.ToFloatTensorInZeroOne(),         # THWC uint8 -> CTHW float in [0,1]
        T.Resize((128, 171)),
        normalize,
        T.CenterCrop((112, 112)),
    ])


def _window_indices(t: int, n: int):
    start = t - WIN // 2
    return [min(max(start + i, 0), n - 1) for i in range(WIN)]


@torch.no_grad()
def extract_clip(frames_rgb, model, transform, device, batch_size: int) -> np.ndarray:
    """Return per-frame features (N, 512) as float32."""
    n = len(frames_rgb)
    frames = np.stack(frames_rgb)           # (N, H, W, 3) uint8
    feats = np.empty((n, 512), dtype=np.float32)
    for s in range(0, n, batch_size):
        ts = range(s, min(s + batch_size, n))
        clips = []
        for t in ts:
            win = torch.from_numpy(frames[_window_indices(t, n)])   # (16, H, W, 3) uint8
            clips.append(transform(win))                            # (3, 16, 112, 112)
        batch = torch.stack(clips).to(device, non_blocking=True)    # (B, 3, 16, 112, 112)
        out = model(batch)                                          # (B, 512)
        feats[s:s + len(clips)] = out.float().cpu().numpy()
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--batch-size", type=int, default=32, help="frames (windows) per forward pass")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    assert os.path.exists(args.ckpt), f"checkpoint not found: {args.ckpt}"
    model = build_backbone(args.ckpt, device)
    transform = build_transform()

    clips = frame_io.list_clips(args.frames_dir)
    clips = frame_io.shard(clips, args.shard_id, args.num_shards)
    if args.limit:
        clips = clips[: args.limit]

    print(f"[tsp] {len(clips)} clips | shard {args.shard_id+1}/{args.num_shards} | "
          f"ckpt={args.ckpt} | out={args.out_dir}")

    t0 = time.time()
    for i, (clip_id, clip_dir) in enumerate(clips):
        if not args.overwrite and frame_io.already_done(args.out_dir, clip_id):
            continue
        frames = frame_io.load_frames_rgb(clip_dir)
        feat = extract_clip(frames, model, transform, device, args.batch_size)   # (N, 512)
        frame_io.save_feature(args.out_dir, clip_id, feat)
        if (i + 1) % 20 == 0 or i == 0:
            dt = time.time() - t0
            print(f"  [{i+1}/{len(clips)}] {clip_id} {tuple(feat.shape)} | {dt/(i+1):.2f}s/clip")
    print(f"[tsp] done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
