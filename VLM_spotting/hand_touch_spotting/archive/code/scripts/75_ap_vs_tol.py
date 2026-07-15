"""AP@delta curve: the E2E eval (mAP) as a function of tolerance delta = 0,1,2,...,10 frames.
y = AP% (touch, untouch, and their mean = mAP). This is the standard action-spotting
'mAP@delta vs delta' curve; the @0/@1/@2 points are exactly the E2E numbers 7.8/45.5/72.9.
CPU-only, uses the cached preds from scripts/73 (no GPU/inference)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_COMMON = "/home/chang_noroot/data2/huyanh/Workspace/hand_touch_detection"
sys.path.insert(0, _COMMON); sys.path.insert(0, os.path.join(_COMMON, "methods/spot_head"))
LAB = os.path.join(_COMMON, "data", "HOI4D-v3")
GRID = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/grid_even"
PRED_CACHE = os.path.join(ROOT, "plot", "precision_preds.json")
TOLS = list(range(0, 11))                                         # delta = 0..10 frames


def main():
    vids = [v for v in json.load(open(os.path.join(LAB, "test.json")))
            if os.path.exists(os.path.join(GRID, v["video"].replace("/", "__") + ".npy"))]
    pr = json.load(open(PRED_CACHE))
    truth = [{"video": v["video"], "events": [{"label": e["label"], "frame": e["frame"]} for e in v["events"]]}
             for v in vids]
    predl = [{"video": v["video"], "events": pr[v["video"]]} for v in vids]

    from common.score import parse_ground_truth, get_predictions, compute_average_precision
    tbl = parse_ground_truth(truth)
    classes = sorted(tbl.keys())                                  # ['touch','untouch']
    curves = {lab: [] for lab in classes}; mean_curve = []
    for t in TOLS:
        per = []
        for lab in classes:
            ap = compute_average_precision(get_predictions(predl, label=lab), tbl[lab], tolerance=t) * 100
            curves[lab].append(ap); per.append(ap)
        mean_curve.append(float(np.mean(per)))
    hdr = "  delta:  " + "".join(f"{t:>6}" for t in TOLS)
    print(hdr, flush=True)
    for lab in classes:
        print(f"  {lab:<7}" + "".join(f"{v:>6.1f}" for v in curves[lab]), flush=True)
    print(f"  {'mAP':<7}" + "".join(f"{v:>6.1f}" for v in mean_curve), flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = {"touch": "#1d7fd1", "untouch": "#e0781e"}
    for lab in classes:
        ax.plot(TOLS, curves[lab], "-o", color=colors[lab], lw=1.8, ms=4, alpha=0.85, label=f"{lab} AP")
    ax.plot(TOLS, mean_curve, "-o", color="#c0392b", lw=2.6, ms=5, label="mAP (E2E eval)")
    for t in (0, 1, 2):                                           # annotate the headline points
        ax.annotate(f"{mean_curve[t]:.1f}", (t, mean_curve[t]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, color="#c0392b", fontweight="bold")
    ax.set_xticks(TOLS); ax.set_xlim(-0.2, TOLS[-1] + 0.2); ax.set_ylim(0, 100)
    ax.set_xlabel("tolerance δ (frames)  [15 fps → 1 frame ≈ 67 ms]")
    ax.set_ylabel("AP  (%)"); ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=10)
    ax.set_title("SFT-LoRA (strongest) — mAP@δ vs tolerance (TEST)  |  E2E: @0 7.8 · @1 45.5 · @2 72.9",
                 fontsize=11)
    fig.tight_layout()
    out = os.path.join(ROOT, "plot", "ap_vs_tol_test.png")
    fig.savefig(out, dpi=120); print(f"[plot] {out}", flush=True)


if __name__ == "__main__":
    main()
