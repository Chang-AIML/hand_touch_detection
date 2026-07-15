"""Re-eval a saved idx checkpoint (no retraining). For M-RoPE checkpoints, pass
--mrope so the prompt gets video-time positions AND generation uses the fixed
manual decode (generated tokens get base_t+step positions matching training)."""
from __future__ import annotations

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/home/chang_noroot/data2/hf_cache")
FEAT = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/feat_interleave"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mrope", action="store_true")
    ap.add_argument("--anchor_stride", type=int, default=5)
    ap.add_argument("--anchor_max_side", type=int, default=252)
    ap.add_argument("--split", default="val")
    ap.add_argument("--eval_videos", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=6)
    args = ap.parse_args()

    from models.wrapper import QwenWrapper
    from models.vjepa_adaptor import VJEPAAdaptor
    from models.idx_localizer import IdxLocalizer
    from eval.idx_metrics import evaluate_multievent_idx

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    rank = cfg.get("lora_rank", 16)
    W = QwenWrapper(device="cuda", dtype=torch.bfloat16)
    ad = VJEPAAdaptor(768, W.d_llm, hidden=2048, n_layers=2).to("cuda", torch.bfloat16)
    ad.set_target_rms_from(W.embed_tokens.weight)     # NOT in state_dict -> must restore!
    ad.load_state_dict(ck["adaptor"]); ad.eval()
    if cfg.get("lora") and "lora" in ck:
        lp = W.add_lora(rank=rank, alpha=2 * rank, n_layers=W.n_layers, target="all")
        for p, saved in zip(lp, ck["lora"]):
            p.data.copy_(saved.to(p.device, p.dtype))
    loz = IdxLocalizer(W, ad, use_idx=cfg.get("use_idx", True), use_anchor=True,
                       anchor_stride=args.anchor_stride, anchor_max_side=args.anchor_max_side,
                       use_mrope=args.mrope, fps=15, max_frames=320, grad_checkpoint=False)
    res = evaluate_multievent_idx(loz, args.split, FEAT, fps=15, batch_size=args.batch_size,
                                  question_phase="test",
                                  types=tuple(cfg.get("types", ["touch", "untouch"])),
                                  max_videos=args.eval_videos, log_every=20)
    tag = f"{os.path.basename(os.path.dirname(args.ckpt))} mrope={args.mrope}"
    print(f"RESULT | {tag} | split={args.split} | "
          f"mAP@0/1/2/4 = {[round(res[f'mAP@{t}'],2) for t in (0,1,2,4)]} "
          f"| @012 {round(res['mAP_012'],2)} | ndet {round(res['_avg_ndet'],2)} "
          f"vs ngt {round(res['_avg_ngt'],2)}", flush=True)


if __name__ == "__main__":
    main()
