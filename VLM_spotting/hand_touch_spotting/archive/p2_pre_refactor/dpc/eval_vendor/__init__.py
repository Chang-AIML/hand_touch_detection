"""Vendored spotting-mAP scorer (self-contained copy of the relevant modules from
the external `hand_touch_detection` repo).

Original sources (staged at /home/chang/Project/vlm_deps/hand_touch_detection):
  common/score.py            -> dpc/eval_vendor/score.py
  common/eval.py             -> dpc/eval_vendor/eval.py   (NMS + soft-NMS)
  common/io.py               -> dpc/eval_vendor/io.py     (json / gz-json loaders, used by the CLI)
  methods/spot_head/eval_nms.py -> dpc/eval_vendor/eval_nms.py

This lets the training/eval code import the scorer WITHOUT the external repo on
sys.path:

    from dpc.eval_vendor.score import compute_mAPs
    from dpc.eval_vendor.eval_nms import maps_quiet

Only the scorer + NMS post-processing is vendored (numpy, optional tabulate).
No heavy deps (torch/models/datasets) are pulled in.
"""
from .score import (  # noqa: F401
    compute_mAPs,
    compute_average_precision,
    parse_ground_truth,
    get_predictions,
)
from .eval_nms import maps_quiet, per_class_aps, TOLS  # noqa: F401
from .eval import (  # noqa: F401
    non_maximum_supression,
    soft_non_maximum_supression,
)

__all__ = [
    "compute_mAPs",
    "compute_average_precision",
    "parse_ground_truth",
    "get_predictions",
    "maps_quiet",
    "per_class_aps",
    "TOLS",
    "non_maximum_supression",
    "soft_non_maximum_supression",
]
