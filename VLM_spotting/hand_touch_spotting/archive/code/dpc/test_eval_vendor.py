#!/usr/bin/env python3
"""Smoke test for the vendored scorer (dpc/eval_vendor).

Runs WITHOUT the external hand_touch_detection repo on sys.path. Constructs a
tiny synthetic (truth, pred) set in the exact format the scorer expects and
exercises:
    dpc.eval_vendor.score.compute_mAPs
    dpc.eval_vendor.eval_nms.maps_quiet
    dpc.eval_vendor.eval.{non_maximum_supression, soft_non_maximum_supression}

Run:  python -m dpc.test_eval_vendor        (from HTS repo root)
   or python dpc/test_eval_vendor.py
"""
from __future__ import annotations

import os
import sys

# make `dpc` importable (HTS repo root = parent of this file's dir), and make
# sure NO external hand_touch_detection copy leaks onto sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path[:] = [p for p in sys.path if "hand_touch_detection" not in p]

from dpc.eval_vendor.score import compute_mAPs                       # noqa: E402
from dpc.eval_vendor.eval_nms import maps_quiet, per_class_aps       # noqa: E402
import dpc.eval_vendor.eval_nms as en                                # noqa: E402
from dpc.eval_vendor.eval import (non_maximum_supression,            # noqa: E402
                                  soft_non_maximum_supression)


def build_synthetic():
    """Two videos, two labels (touch/untouch).

    truth events : {"video", "events":[{"label","frame"}, ...]}
    pred events  : {"video", "events":[{"label","frame","score"}, ...]}
    Predictions include exact hits, near-hits (within tol), and a distractor.
    """
    truth = [
        {"video": "vidA",
         "events": [{"label": "touch", "frame": 100},
                    {"label": "untouch", "frame": 200}]},
        {"video": "vidB",
         "events": [{"label": "touch", "frame": 50},
                    {"label": "touch", "frame": 80}]},
    ]
    pred = [
        {"video": "vidA",
         "events": [{"label": "touch", "frame": 100, "score": 0.95},   # exact
                    {"label": "untouch", "frame": 202, "score": 0.90},  # off by 2
                    {"label": "touch", "frame": 130, "score": 0.40}]},  # distractor
        {"video": "vidB",
         "events": [{"label": "touch", "frame": 51, "score": 0.85},    # off by 1
                    {"label": "touch", "frame": 79, "score": 0.70},    # off by 1
                    {"label": "touch", "frame": 55, "score": 0.30}]},  # near-dup distractor
    ]
    return truth, pred


def main():
    truth, pred = build_synthetic()

    print("=== compute_mAPs (prints per-class table) ===")
    mAPs, tols = compute_mAPs(truth, pred, tolerances=[0, 1, 2, 4])
    print("returned mAPs (fraction 0..1):", [round(m, 4) for m in mAPs])
    print("tolerances:", tols)
    print("mAP@{0,1,2,4} (%):",
          {f"mAP@{t}": round(float(mAPs[i]) * 100, 2) for i, t in enumerate(tols)})

    print("\n=== maps_quiet (swallows table, returns %) ===")
    en.TOLS = [0, 1, 2, 4]
    m_pct = maps_quiet(truth, pred)
    print("maps_quiet ->", [round(x, 2) for x in m_pct])

    print("\n=== maps_quiet with TOLS=[0,1,2] (as idx_compress_train sets) ===")
    en.TOLS = [0, 1, 2]
    m3 = maps_quiet(truth, pred)
    print("maps_quiet(TOLS=[0,1,2]) ->", [round(x, 2) for x in m3])

    print("\n=== per_class_aps ===")
    en.TOLS = [0, 1, 2, 4]
    for label, aps in per_class_aps(truth, pred).items():
        print(f"  {label}: {[round(a, 2) for a in aps]}")

    print("\n=== NMS post-processings ===")
    nms = non_maximum_supression(pred, window=2)
    snms = soft_non_maximum_supression(pred, window=4)
    print("hard-NMS vidB events:", [(e["frame"], round(e["score"], 2)) for e in nms[1]["events"]])
    print("soft-NMS vidB events:", [(e["frame"], round(e["score"], 2)) for e in snms[1]["events"]])
    en.TOLS = [0, 1, 2, 4]
    print("maps_quiet after hard-NMS ->", [round(x, 2) for x in maps_quiet(truth, nms)])

    print("\nOK: vendored scorer ran self-contained (no external repo on sys.path).")


if __name__ == "__main__":
    main()
