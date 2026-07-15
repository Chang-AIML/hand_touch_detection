"""Phase 1 evaluation on a split from a saved checkpoint -> AP@{0,1,2,4}+MAE [GATE 2].

  python scripts/21_phase1_eval.py --ckpt outputs/phase1/<run>/best.pt --split test
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/home/chang_noroot/data2/hf_cache")

# V-JEPA→MS-TCN (interleave) baseline for GATE 2 (mAP %, test)
VJEPA_BASELINE = {"mAP@0": 19.66, "mAP@1": 49.83, "mAP@2": 67.84, "mAP@4": 81.28,
                  "mAP@2_softnms": 72.14}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--hard_argmax", action="store_true")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    from train.train_loop import build
    from data.dataset import Phase1Dataset
    from eval.metrics import evaluate

    W, ad, loc, loz = build(cfg, "cuda")
    ad.load_state_dict(ckpt["adaptor"]); loc.load_state_dict(ckpt["loc"])
    if loz.sim_head is not None and "sim_head" in ckpt:
        loz.sim_head.load_state_dict(ckpt["sim_head"])
    if loz.film is not None and "film" in ckpt:
        loz.film.load_state_dict(ckpt["film"])
    ad.eval()
    ds = Phase1Dataset(args.split, cfg["feat_dir"])
    print(f"[eval] {args.split}: {len(ds)} samples | ckpt epoch {ckpt.get('epoch')} "
          f"| sim_layer {cfg['sim_layer']} anchor {cfg['use_anchor']}")
    res = evaluate(loz, ds, tolerances=cfg["tolerances"], batch_size=args.batch_size,
                   use_soft_argmax=not args.hard_argmax)
    print("\n=== Phase 1 test results ===")
    print(f"  MAE (frames): all {res['mae']['all']:.2f} | "
          f"touch {res['mae'].get('touch', float('nan')):.2f} | "
          f"untouch {res['mae'].get('untouch', float('nan')):.2f}")
    for k in ("mAP@0", "mAP@1", "mAP@2", "mAP@4", "mAP_012"):
        print(f"  {k}: {res['ap'][k]:.2f}")
    print("\n=== GATE 2 vs V-JEPA+MSTCN baseline (interleave) ===")
    print(f"  ours mAP@2 {res['ap']['mAP@2']:.2f}  vs  baseline no-NMS {VJEPA_BASELINE['mAP@2']:.2f} "
          f"/ soft-NMS {VJEPA_BASELINE['mAP@2_softnms']:.2f}")
    verdict = "GO (>= no-NMS baseline)" if res['ap']['mAP@2'] >= VJEPA_BASELINE['mAP@2'] else \
              "below no-NMS baseline — inspect (time_index / sim_layer / temp)"
    print(f"  GATE 2: {verdict}")


if __name__ == "__main__":
    main()
