"""The AP curve = the plot whose AREA is the E2E eval. Reproduces common.score's
compute_average_precision exactly (sort preds by score, one-to-one nearest-unmatched-GT
match, interpolated precision, AP = sum(interp)/#GT) and fills the area so you SEE that
area = AP. Per tolerance {0,1,2}, per class {touch,untouch}; the class-mean = mAP@tol =
the E2E number (7.8/45.5/72.9). Uses the cached preds from scripts/73 -> no GPU/inference."""
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
import eval_nms as en                                             # noqa: E402


def pr_and_ap(flat_pred, truth_by_video, tolerance, total):
    """Exact replica of common.score.compute_average_precision (returns the curve too)."""
    recalled = set(); pc = []
    for i, (video, frame, score) in enumerate(flat_pred, 1):
        gt_closest = None
        for gt_frame in truth_by_video.get(video, []):
            if (video, gt_frame) in recalled:
                continue
            if gt_closest is None or abs(frame - gt_closest) > abs(frame - gt_frame):
                gt_closest = gt_frame
        if gt_closest is not None and abs(frame - gt_closest) <= tolerance:
            recalled.add((video, gt_closest)); pc.append(len(recalled) / i)
    interp, mx = [], 0.0
    for p in pc[::-1]:
        mx = max(p, mx); interp.append(mx)
    interp.reverse()
    ap = sum(interp) / total if total else 0.0
    rc = [(k + 1) / total for k in range(len(pc))]                # recall at each TP
    return rc, pc, interp, ap * 100


def main():
    vids = [v for v in json.load(open(os.path.join(LAB, "test.json")))
            if os.path.exists(os.path.join(GRID, v["video"].replace("/", "__") + ".npy"))]
    pr = json.load(open(PRED_CACHE))
    truth = [{"video": v["video"], "events": [{"label": e["label"], "frame": e["frame"]} for e in v["events"]]}
             for v in vids]
    predl = [{"video": v["video"], "events": pr[v["video"]]} for v in vids]
    en.TOLS = [0, 1, 2]
    map_check = en.maps_quiet(truth, predl)                       # exact E2E numbers
    print(f"[E2E maps_quiet] mAP@0/1/2 = {[round(x,2) for x in map_check]}", flush=True)

    from common.score import parse_ground_truth, get_predictions
    tbl = parse_ground_truth(truth)                              # label -> {video: [frames]}
    classes = sorted(tbl.keys())
    colors = {"touch": "#1d7fd1", "untouch": "#e0781e"}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ci, tol in enumerate([0, 1, 2]):
        ax = axes[ci]; aps = []
        for lab in classes:
            total = sum(len(f) for f in tbl[lab].values())
            rc, pc, interp, ap = pr_and_ap(get_predictions(predl, label=lab), tbl[lab], tol, total)
            aps.append(ap)
            # step curve to recall_max, then drop to 0 (missed GT -> precision 0), fill = AP
            xr = [0] + rc + [rc[-1] if rc else 0, 1.0]
            yr = [interp[0] if interp else 0] + interp + [0, 0]
            ax.step(xr, yr, where="post", color=colors[lab], lw=2.0,
                    label=f"{lab}  AP={ap:.1f}")
            ax.fill_between(xr, yr, step="post", color=colors[lab], alpha=0.15)
        mean_ap = float(np.mean(aps))
        ax.set_title(f"tolerance @{tol}  —  mAP = {mean_ap:.1f}", fontsize=11)
        ax.set_xlabel("recall"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=9)
        if ci == 0:
            ax.set_ylabel("precision")
        ax.text(0.03, 0.06, f"area = AP\n(E2E mAP@{tol}={map_check[tol]:.1f})", transform=ax.transAxes,
                fontsize=8.5, family="monospace", va="bottom",
                bbox=dict(boxstyle="round", fc="#f5f5f5", ec="gray"))
    fig.suptitle("SFT-LoRA (strongest) — Precision-Recall curves, AREA = AP = the E2E eval "
                 "(TEST).  mAP@0/1/2 = 7.8 / 45.5 / 72.9", fontsize=12)
    fig.tight_layout()
    out = os.path.join(ROOT, "plot", "pr_curve_test.png")
    fig.savefig(out, dpi=110); print(f"[plot] {out}", flush=True)
    print("  per-class/tol APs computed above; class-mean per panel = the mAP shown.", flush=True)


if __name__ == "__main__":
    main()
