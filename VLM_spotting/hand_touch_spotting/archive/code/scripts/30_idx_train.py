"""CLI for idx-decode training (single GPU; pin with CUDA_VISIBLE_DEVICES)."""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--no_idx", action="store_true", help="drop per-frame index handles (count, not copy)")
    ap.add_argument("--no_anchor", action="store_true", help="drop ViT anchors")
    ap.add_argument("--anchor_stride", type=int, default=1, help="place a ViT anchor every N seconds")
    ap.add_argument("--anchor_max_side", type=int, default=168, help="anchor ViT resolution (252=original-ish)")
    ap.add_argument("--mrope", action="store_true", help="M-RoPE: video-time 3-channel position_ids")
    ap.add_argument("--multi_event", action="store_true", help="HONEST full-video multi-event (mAP vs 41/68)")
    ap.add_argument("--eval_videos", type=int, default=200, help="val videos for per-epoch multi-event mAP")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--lr_adaptor", type=float, default=1e-3)
    ap.add_argument("--lr_lora", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--crop_secs", type=int, default=6)
    ap.add_argument("--train_limit", type=int, default=4000)
    ap.add_argument("--val_limit", type=int, default=600)
    ap.add_argument("--types", default="touch,untouch")
    args = ap.parse_args()

    from train.idx_train import run
    run(use_idx=not args.no_idx, use_anchor=not args.no_anchor,
        anchor_stride=args.anchor_stride, anchor_max_side=args.anchor_max_side,
        use_mrope=args.mrope, lora=args.lora,
        lora_rank=args.lora_rank, lr_adaptor=args.lr_adaptor, lr_lora=args.lr_lora,
        epochs=args.epochs, batch_size=args.batch_size, crop_secs=args.crop_secs,
        train_limit=args.train_limit, val_limit=args.val_limit,
        types=tuple(args.types.split(",")), run_name=args.run_name,
        multi_event=args.multi_event, eval_videos=args.eval_videos)


if __name__ == "__main__":
    main()
