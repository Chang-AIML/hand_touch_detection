"""Point-event mAP@tolerance — the thin wrapper the training/eval loop calls.
Vendored from hand_touch_detection/methods/spot_head/eval_nms.py; only these two symbols are live.

  TOLS                     module-global tolerances; callers set e.g. en.TOLS = [0,1,2,4,8,16]
  maps_quiet(truth, pred)  per-tolerance mAP as PERCENT (mAP*100), one per TOLS entry,
                           swallowing compute_mAPs' internal per-class table print.
"""
from __future__ import annotations
import contextlib, io
from .score import compute_mAPs

TOLS = [0, 1, 2, 4]


def maps_quiet(truth, pred):
    with contextlib.redirect_stdout(io.StringIO()):              # swallow compute_mAPs' table print
        mAPs, _ = compute_mAPs(truth, pred, tolerances=TOLS)
    return [m * 100 for m in mAPs]
