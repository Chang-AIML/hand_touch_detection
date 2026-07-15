"""CLI: train the Action Head (δ-offset regressor) on top of the frozen idx-gen model."""
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
    ap.add_argument("--run_name", default="action_head")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pool_videos", type=int, default=800)
    ap.add_argument("--nms_win", type=int, default=2)
    ap.add_argument("--huber_beta", type=float, default=1.0)
    args = ap.parse_args()

    from train.action_train import run
    run(gen_ckpt=args.gen_ckpt, W_win=args.window, d=args.d, n_heads=args.n_heads,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        pool_videos=args.pool_videos, nms_win=args.nms_win,
        run_name=args.run_name, huber_beta=args.huber_beta)


if __name__ == "__main__":
    main()
