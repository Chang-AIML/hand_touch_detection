"""Load a checkpoint (VLM or probe) and run evaluate_multievent under a given config
(split / categories / exclude / types / question_phase). Used for the Q1/Q2/Q3 matrix."""
from __future__ import annotations

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/home/chang_noroot/data2/hf_cache")


def load_model(ckpt_path, device="cuda"):
    from train.train_loop import build
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    W, ad, loc, loz = build(cfg, device)
    if loz.__class__.__name__ == "ProbeModel":
        loz.load_state_dict(ck["probe"])
    else:
        ad.load_state_dict(ck["adaptor"]); loc.load_state_dict(ck["loc"])
        if loz.sim_head is not None and "sim_head" in ck: loz.sim_head.load_state_dict(ck["sim_head"])
        if loz.film is not None and "film" in ck: loz.film.load_state_dict(ck["film"])
        if getattr(loz, "lora_params", []) and "lora" in ck:
            for p, saved in zip(loz.lora_params, ck["lora"]):
                p.data.copy_(saved.to(p.device, p.dtype))
    ad.eval()
    return cfg, loz


def parse_cats(s):
    return None if not s else [int(c) for c in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--categories", default="")
    ap.add_argument("--exclude_cats", default="")
    ap.add_argument("--types", default="touch,untouch")
    ap.add_argument("--question_phase", default="canonical", choices=["canonical", "test", "train"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from eval.metrics import evaluate_multievent
    cfg, loz = load_model(args.ckpt)
    ap_res = evaluate_multievent(
        loz, args.split, cfg["feat_dir"], fps=cfg["fps"],
        window_secs=cfg.get("crop_secs", 10), stride_secs=max(1, cfg.get("crop_secs", 10) // 2),
        tolerances=[0, 1, 2, 4], batch_size=24,
        min_gap=cfg.get("nms_min_gap", 15), rel_thresh=cfg.get("nms_rel_thresh", 0.5),
        categories=parse_cats(args.categories), exclude_categories=parse_cats(args.exclude_cats),
        types=tuple(args.types.split(",")), question_phase=args.question_phase)
    tag = args.tag or os.path.basename(os.path.dirname(args.ckpt))
    print(f"RESULT | {tag} | split={args.split} cats={args.categories or 'all'} "
          f"excl={args.exclude_cats or '-'} types={args.types} q={args.question_phase} "
          f"| mAP@0/1/2/4 = {[round(ap_res[f'mAP@{t}'],2) for t in (0,1,2,4)]} | @012 {round(ap_res['mAP_012'],2)}")


if __name__ == "__main__":
    main()
