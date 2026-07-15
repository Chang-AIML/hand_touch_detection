"""CLI: train a stage-2 language-conditioned SALIENCY refiner (tcn | asformer)."""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_ckpt", default=os.path.join(ROOT, "outputs/idx/idx_multi/best.pt"))
    ap.add_argument("--run_name", default="refine")
    ap.add_argument("--head_type", default="tcn", choices=["tcn", "asformer"])
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--pool_videos", type=int, default=800)
    ap.add_argument("--nms_win", type=int, default=2)
    ap.add_argument("--sigma", type=float, default=1.5)
    ap.add_argument("--lam_l1", type=float, default=0.5)
    ap.add_argument("--query_mode", default="grounded", choices=["grounded", "cfree"])
    args = ap.parse_args()

    from train.refine_train import run
    run(gen_ckpt=args.gen_ckpt, head_type=args.head_type, W_win=args.window, d=args.d,
        n_layers=args.n_layers, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        pool_videos=args.pool_videos, nms_win=args.nms_win, sigma=args.sigma,
        lam_l1=args.lam_l1, query_mode=args.query_mode, run_name=args.run_name)


if __name__ == "__main__":
    main()
