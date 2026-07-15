"""Phase 1 training CLI — thin wrapper over train.train_loop.run."""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "train.yaml"))
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--sim_layer", type=int)
    ap.add_argument("--use_anchor", action="store_true")
    ap.add_argument("--run_name")
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="1 tiny epoch on a few samples")
    ap.add_argument("--probe", action="store_true", help="no-VLM probe baseline")
    ap.add_argument("--exclude_cats", help="comma list of held-out object categories (Q2)")
    ap.add_argument("--train_types", help="comma list, e.g. 'touch' for zero-shot untouch (Q3)")
    ap.add_argument("--lora", action="store_true", help="LoRA-unfreeze the first sim_layer LLM layers")
    ap.add_argument("--align_target", choices=["adaptor", "postllm"], help="LOC aligns to")
    ap.add_argument("--lr_lora", type=float)
    ap.add_argument("--lora_rank", type=int)
    ap.add_argument("--no_grad_ckpt", action="store_true", help="disable grad checkpointing (use spare VRAM)")
    args = ap.parse_args()

    over = {}
    for k in ("epochs", "batch_size", "sim_layer", "run_name"):
        v = getattr(args, k)
        if v is not None:
            over[k] = v
    if args.use_anchor:
        over["use_anchor"] = True
    if args.no_wandb:
        over["wandb"] = False
    if args.probe:
        over["probe_mode"] = True
    if args.exclude_cats:
        over["exclude_categories"] = [int(c) for c in args.exclude_cats.split(",")]
    if args.train_types:
        over["train_types"] = args.train_types.split(",")
    if args.lora:
        over["lora"] = True
    if args.align_target:
        over["align_target"] = args.align_target
    if args.lr_lora is not None:
        over["lr_lora"] = args.lr_lora
    if args.lora_rank is not None:
        over["lora_rank"] = args.lora_rank
        over["lora_alpha"] = 2 * args.lora_rank
    if args.no_grad_ckpt:
        over["grad_checkpoint"] = False

    from train.train_loop import run
    run(args.config, **over)


if __name__ == "__main__":
    main()
