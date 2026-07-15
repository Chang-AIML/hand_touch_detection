"""Diagnostic #2: full-video score-curve + distribution analysis for a checkpoint.

Saves to results/: 20 example score curves (curve, GT, pred, top-5 peaks) and
distribution figures (GT vs pred frame hist, |error| hist, touch/untouch errors),
plus text stats (systematic bias, multi-peak rate).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/home/chang_noroot/data2/hf_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def top_k_peaks(curve, k=5, min_gap=8):
    order = np.argsort(-curve)
    peaks = []
    for idx in order:
        if all(abs(idx - p) >= min_gap for p in peaks):
            peaks.append(int(idx))
        if len(peaks) >= k:
            break
    return peaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=250)
    args = ap.parse_args()

    from train.train_loop import build
    from data.dataset import Phase1Dataset
    from eval.metrics import predict_windowed
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False); cfg = ck["cfg"]
    W, ad, loc, loz = build(cfg, "cuda")
    ad.load_state_dict(ck["adaptor"]); loc.load_state_dict(ck["loc"])
    if loz.sim_head is not None and "sim_head" in ck: loz.sim_head.load_state_dict(ck["sim_head"])
    if loz.film is not None and "film" in ck: loz.film.load_state_dict(ck["film"])
    ad.eval()
    ds = Phase1Dataset(args.split, cfg["feat_dir"], crop_secs=0)
    preds = predict_windowed(loz, ds, window_secs=cfg.get("crop_secs", 10),
                             stride_secs=max(1, cfg.get("crop_secs", 10)//2), fps=cfg["fps"],
                             batch_size=24, max_samples=args.n, return_curves=True)
    outdir = os.path.join(ROOT, "results", "curves"); os.makedirs(outdir, exist_ok=True)

    gts = np.array([p["gt"] for p in preds]); pfs = np.array([p["pred_frame"] for p in preds])
    err = np.abs(pfs - gts); signed = pfs - gts
    multi = 0
    for p in preds:
        pk = top_k_peaks(p["curve"])
        if len(pk) >= 2 and (p["curve"][pk[1]] > 0.8 * p["curve"][pk[0]]):
            multi += 1
    print(f"=== curve diag {args.ckpt} (epoch {ck.get('epoch')}, n={len(preds)}) ===")
    print(f"  |error| mean {err.mean():.1f} median {np.median(err):.1f} | signed-error mean "
          f"{signed.mean():+.1f} (systematic bias)")
    print(f"  hit@ <=2: {100*(err<=2).mean():.1f}%  <=4: {100*(err<=4).mean():.1f}%  "
          f"<=15(~1s): {100*(err<=15).mean():.1f}%")
    print(f"  multi-peak (2nd peak > 0.8x top): {100*multi/len(preds):.1f}% of videos")
    print(f"  GT frames: mean {gts.mean():.0f} std {gts.std():.0f} | "
          f"PRED frames: mean {pfs.mean():.0f} std {pfs.std():.0f}")

    # distribution figure
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].hist(gts, bins=30, alpha=.6, label="GT"); ax[0, 0].hist(pfs, bins=30, alpha=.6, label="pred")
    ax[0, 0].legend(); ax[0, 0].set_title("GT vs predicted frame distribution")
    ax[0, 1].hist(err, bins=40); ax[0, 1].set_title("|error| histogram (frames)")
    te = err[[i for i, p in enumerate(preds) if p["type"] == "touch"]]
    ue = err[[i for i, p in enumerate(preds) if p["type"] == "untouch"]]
    ax[1, 0].hist(te, bins=30); ax[1, 0].set_title(f"touch |error| (mean {te.mean():.1f})")
    ax[1, 1].hist(ue, bins=30); ax[1, 1].set_title(f"untouch |error| (mean {ue.mean():.1f})")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "distributions.png"), dpi=90); plt.close(fig)

    # 20 example curves
    fig, axes = plt.subplots(5, 4, figsize=(20, 14))
    for i, (p, axx) in enumerate(zip(preds[:20], axes.flat)):
        c = p["curve"]; axx.plot(c, lw=.8)
        axx.axvline(p["gt"], color="g", ls="--", label="GT")
        axx.axvline(p["pred_frame"], color="r", ls=":", label="pred")
        for pk in top_k_peaks(c): axx.plot(pk, c[pk], "k.", ms=4)
        axx.set_title(f"{p['type']} gt{p['gt']} pred{int(p['pred_frame'])}", fontsize=8)
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "example_curves.png"), dpi=80); plt.close(fig)
    print(f"  saved -> {outdir}/distributions.png , example_curves.png")


if __name__ == "__main__":
    main()
