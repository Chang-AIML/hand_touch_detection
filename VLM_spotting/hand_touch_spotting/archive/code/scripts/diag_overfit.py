"""Decisive diagnostic: can the localizer OVERFIT a small train subset to low MAE?

If yes -> the readout mechanism works, tune for generalisation. If train MAE stays
~random (~68 frames on 300-frame clips), the readout/loss is fundamentally broken.
"""
from __future__ import annotations

import os
import sys
import argparse

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/home/chang_noroot/data2/hf_cache")

from train.train_loop import build, trainable_params        # noqa: E402
from train.loss import loc_loss                             # noqa: E402
from data.dataset import Phase1Dataset                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=None)
    ap.add_argument("--label_smooth", type=float, default=None)
    ap.add_argument("--adaptor_only", action="store_true",
                    help="train ONLY the input adaptor (freeze LOC, no output head)")
    ap.add_argument("--align_target", default=None, choices=[None, "postllm", "adaptor"])
    ap.add_argument("--anchor", dest="anchor", action="store_true")
    ap.add_argument("--lambda_mae", type=float, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "train.yaml")))
    if args.align_target is not None:
        cfg["align_target"] = args.align_target
    if args.anchor:
        cfg["use_anchor"] = True
        cfg["anchor_max_side"] = 252          # cheap ViT anchors (~49 tok/anchor)
    if args.lambda_mae is not None:
        cfg["lambda_mae"] = args.lambda_mae
    if args.temp is not None:
        cfg["temp"] = args.temp
    if args.label_smooth is not None:
        cfg["label_smooth_frames"] = args.label_smooth
    if args.adaptor_only:
        cfg["sim_head"] = False                       # no output head -> plain cosine readout
    W, ad, loc, loz = build(cfg, "cuda")
    print(f"[cfg] temp={cfg['temp']} label_smooth={cfg['label_smooth_frames']} lambda_mae={cfg['lambda_mae']} "
          f"adaptor_only={args.adaptor_only} sim_head={loz.sim_head is not None}")
    ps = list(ad.parameters()) if args.adaptor_only else trainable_params(ad, loc, loz)
    print(f"[overfit] trainable {sum(p.numel() for p in ps)/1e6:.2f}M")
    opt = torch.optim.AdamW(ps, lr=args.lr, weight_decay=0.0)

    ds = Phase1Dataset("train", cfg["feat_dir"])
    idx = list(range(args.n))
    samples = [ds[i] for i in idx]
    loz.train(); ad.train()
    for step in range(args.steps):
        bi = np.random.choice(args.n, args.bs, replace=False)
        batch = [samples[i] for i in bi]
        s_list, metas = loz.forward_batch(batch)
        loss = 0.0
        for s, m in zip(s_list, metas):
            l, _ = loc_loss(s, m["gt"], lambda_mae=cfg["lambda_mae"],
                            label_smooth_frames=cfg["label_smooth_frames"])
            loss = loss + l
        loss = loss / len(s_list)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ps, 1.0)
        opt.step()
        if step % 50 == 0 or step == args.steps - 1:
            # train MAE over the whole subset
            loz.eval()
            with torch.no_grad():
                maes, hits = [], 0
                for j in range(0, args.n, args.bs):
                    b = samples[j:j + args.bs]
                    sl, mm = loz.forward_batch(b)
                    for s, m in zip(sl, mm):
                        sn = s.float().cpu().numpy()
                        e = np.exp(sn - sn.max()); p = e / e.sum()
                        pf = (p * np.arange(len(sn))).sum()
                        maes.append(abs(pf - m["gt"])); hits += int(abs(pf - m["gt"]) <= 2)
            loz.train()
            print(f"  step {step:3d} loss {loss.item():.4f} | train MAE {np.mean(maes):.2f} "
                  f"| @2 acc {hits/args.n*100:.1f}%", flush=True)
    print("[overfit] done — if MAE << 68 and @2 high, the readout works.")


if __name__ == "__main__":
    main()
