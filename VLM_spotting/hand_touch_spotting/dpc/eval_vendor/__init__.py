"""Vendored point-event mAP scorer. `score.py` is the untouchable numeric core;
`eval_nms.maps_quiet` is the thin wrapper the eval loop calls.
    from dpc.eval_vendor.score import compute_mAPs
    from dpc.eval_vendor.eval_nms import maps_quiet, TOLS
"""
from .score import compute_mAPs, compute_average_precision, parse_ground_truth, get_predictions  # noqa: F401
from .eval_nms import maps_quiet, TOLS  # noqa: F401

__all__ = ["compute_mAPs", "compute_average_precision", "parse_ground_truth",
           "get_predictions", "maps_quiet", "TOLS"]
