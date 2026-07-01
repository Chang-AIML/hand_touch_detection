"""Per-frame feature extraction with V-JEPA 2 (vitl-fpc16-256).

Problem: V-JEPA's patch embedding uses tubelet_size=2, so a 16-frame clip yields
only 8 temporal tokens (each token = 2 consecutive frames). To recover a feature
for *every* frame without changing or retraining the model, we run the encoder
TWICE with a 1-frame temporal offset and interleave the results (method "B"):

    even pass:  frames [0,1,2,...]      -> tubelet tokens cover (0,1)(2,3)... -> frames 0,2,4,...
    odd  pass:  frames [1,2,3,...]      -> tubelet tokens cover (1,2)(3,4)... -> frames 1,3,5,...
    interleave: out[2k]=even[k], out[2k+1]=odd[k]

Within each pass we use NON-overlapping 16-frame windows. Each pass is padded up
to a multiple of 16 by repeating the last frame (e.g. [...,3,3,3]). Token k of a
window covers window-local frames (2k, 2k+1) -- this is fixed by the Conv3d in the
patch embedder (kernel/stride temporal = tubelet_size = 2).

Output per clip: a spatial-temporal feature of shape (N, 16, 16, 1024) by default
(--pool none, keeps spatial info as requested). Use frame_io.pool_features(...) at
load time to get temporal-only (N, 1024), or pass --pool temporal to save that
directly.

Run with the `dev` conda env:
    /data/dong/miniconda3/envs/dev/bin/python extract_vjepa.py \
        --frames-dir /data/dong/project/Workspace/dataset/hoi4d/frames \
        --out-dir    ./output/vjepa
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from transformers import VJEPA2Model, AutoVideoProcessor

import frame_io

MODEL_ID = "facebook/vjepa2-vitl-fpc16-256-ssv2"
TUBELET = 2
WIN = 16                      # frames_per_clip the encoder was trained with
TOK_T = WIN // TUBELET        # 8 temporal tokens per window
GRID = 16                     # 256 / patch(16) -> 16x16 spatial tokens
DIM = 1024


def _pad_to_multiple(x: torch.Tensor, m: int) -> torch.Tensor:
    """Pad axis-0 up to a multiple of m by repeating the last frame."""
    n = x.shape[0]
    pad = (-n) % m
    if pad:
        x = torch.cat([x, x[-1:].expand(pad, *x.shape[1:])], dim=0)
    return x


@torch.no_grad()
def _run_pass(frames: torch.Tensor, model, device, batch_windows: int) -> torch.Tensor:
    """frames: (M, 3, 256, 256) preprocessed, M will be padded to a multiple of 16.

    Returns (T, 16, 16, 1024) temporal tokens in time order, T = padded_M / 2.
    Token p maps to local frame 2p (the first frame of its tubelet).
    """
    frames = _pad_to_multiple(frames, WIN)
    nwin = frames.shape[0] // WIN
    wins = frames.view(nwin, WIN, 3, 256, 256)
    chunks = []
    for s in range(0, nwin, batch_windows):
        b = wins[s:s + batch_windows].to(device, torch.float16, non_blocking=True)
        h = model(pixel_values_videos=b).last_hidden_state           # (bw, 2048, 1024)
        bw = h.shape[0]
        h = h.view(bw, TOK_T, GRID, GRID, DIM)                        # (bw, 8, 16, 16, 1024)
        chunks.append(h.float().cpu())
    h = torch.cat(chunks, dim=0)                                      # (nwin, 8, 16, 16, 1024)
    return h.reshape(nwin * TOK_T, GRID, GRID, DIM)                   # (T, 16, 16, 1024)


@torch.no_grad()
def extract_clip(frames_rgb, model, processor, device, batch_windows: int) -> np.ndarray:
    """Return per-frame spatial-temporal features (N, 16, 16, 1024) as float32."""
    n = len(frames_rgb)
    # Preprocess every frame once (shortest-edge 292 -> center crop 256 + ImageNet norm).
    proc = processor(frames_rgb, return_tensors="pt")["pixel_values_videos"][0]  # (N,3,256,256)

    even_tok = _run_pass(proc, model, device, batch_windows)          # frames 0,2,4,...
    odd_tok = _run_pass(proc[1:], model, device, batch_windows)       # frames 1,3,5,...

    out = torch.empty(n, GRID, GRID, DIM, dtype=torch.float32)
    ev = torch.arange(0, n, 2)
    od = torch.arange(1, n, 2)
    out[ev] = even_tok[: ev.numel()]
    out[od] = odd_tok[: od.numel()]
    return out.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pool", choices=["none", "temporal"], default="none",
                    help="none -> save spatial grid (N,16,16,1024); temporal -> save (N,1024)")
    ap.add_argument("--batch-windows", type=int, default=8, help="windows per forward pass")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="debug: only process first N clips of the shard")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
    model = VJEPA2Model.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device).eval()

    clips = frame_io.list_clips(args.frames_dir)
    clips = frame_io.shard(clips, args.shard_id, args.num_shards)
    if args.limit:
        clips = clips[: args.limit]

    print(f"[vjepa] {len(clips)} clips | shard {args.shard_id+1}/{args.num_shards} | "
          f"pool={args.pool} | out={args.out_dir}")
    if args.pool == "none":
        print("[vjepa] NOTE: saving spatial grid (N,16,16,1024) fp16 ~= 0.5MB/frame "
              "(~660GB for the full 4022 clips). Use --pool temporal for ~2.6GB total.")

    t0 = time.time()
    for i, (clip_id, clip_dir) in enumerate(clips):
        if not args.overwrite and frame_io.already_done(args.out_dir, clip_id):
            continue
        frames = frame_io.load_frames_rgb(clip_dir)
        feat = extract_clip(frames, model, processor, device, args.batch_windows)  # (N,16,16,1024)
        if args.pool == "temporal":
            feat = frame_io.pool_features(feat, "temporal")                        # (N,1024)
        frame_io.save_feature(args.out_dir, clip_id, feat)
        if (i + 1) % 20 == 0 or i == 0:
            dt = time.time() - t0
            print(f"  [{i+1}/{len(clips)}] {clip_id} {tuple(feat.shape)} "
                  f"| {dt/(i+1):.2f}s/clip")
    print(f"[vjepa] done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
