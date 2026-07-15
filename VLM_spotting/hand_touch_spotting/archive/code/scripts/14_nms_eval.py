"""Multi-event eval with NMS sweep (matches the baseline's protocol exactly).

Dense per-frame detections (score = sigmoid(aggregated s(t))) -> {none / hard-NMS /
soft-NMS / peak-pick} -> shared frame-tolerance mAP. Reports mAP@{0,1,2,4} + Avg,
touch/untouch, vs the V-JEPA->MSTCN baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/home/chang_noroot/data2/hf_cache")
sys.path.insert(0, "/home/chang_noroot/data2/huyanh/Workspace/hand_touch_detection")
from common.score import compute_mAPs                       # noqa: E402
from common.eval import non_maximum_supression, soft_non_maximum_supression  # noqa: E402

# V-JEPA->MSTCN (interleave) test baseline
BASE = {"none": [19.66, 49.83, 67.84, 81.28], "soft": [17.76, 52.59, 72.14, 86.48]}
TOLS = [0, 1, 2, 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--dense_thresh", type=float, default=0.05, help="min sigmoid score to keep a frame")
    args = ap.parse_args()

    from train.train_loop import build
    from eval.metrics import predict_windowed
    from data.dataset import GENERIC_Q
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False); cfg = ck["cfg"]
    W, ad, loc, loz = build(cfg, "cuda")
    ad.load_state_dict(ck["adaptor"]); loc.load_state_dict(ck["loc"])
    if loz.sim_head is not None and "sim_head" in ck: loz.sim_head.load_state_dict(ck["sim_head"])
    if loz.film is not None and "film" in ck: loz.film.load_state_dict(ck["film"])
    ad.eval()
    fps = cfg["fps"]; win = cfg.get("crop_secs", 10)

    vids = json.load(open(os.path.join(ROOT, "data", "annotations", f"{args.split}.json")))
    vids = [v for v in vids if os.path.exists(os.path.join(cfg["feat_dir"], v["video_id"] + ".npy"))]

    class _DS:
        def __init__(s, it): s.it = it
        def __len__(s): return len(s.it)
        def __getitem__(s, i):
            v, typ = s.it[i]
            f = np.load(os.path.join(cfg["feat_dir"], v + ".npy")).astype(np.float32)
            return {"feats": torch.from_numpy(f), "question": GENERIC_Q[typ], "gt": 0, "fps": fps,
                    "type": typ, "video_id": v, "num_frames": f.shape[0], "full_num_frames": f.shape[0],
                    "anchor_start_sec": 0, "anchor_num_secs": f.shape[0] // fps}
    items = [(v["video_id"], typ) for v in vids for typ in ("touch", "untouch")
             if any(e["type"] == typ for e in v["events"])]
    print(f"[{args.split}] {len(vids)} videos, {len(items)} (video,type) queries; ckpt epoch {ck.get('epoch')}")
    preds = predict_windowed(loz, _DS(items), window_secs=win, stride_secs=max(1, win // 2),
                             fps=fps, batch_size=24, return_curves=True)

    truth = [{"video": v["video_id"], "events": [{"label": e["type"], "frame": e["frame"]}
                                                 for e in v["events"]]} for v in vids]
    # dense per-frame candidate detections per (video,type)
    dense = {v["video_id"]: [] for v in vids}
    for p in preds:
        prob = 1.0 / (1.0 + np.exp(-p["curve"]))          # sigmoid of aggregated s(t)
        for t in np.where(prob >= args.dense_thresh)[0]:
            dense[p["video_id"]].append({"label": p["type"], "frame": int(t), "score": float(prob[t])})
    order = sorted(dense)
    pred_dense = [{"video": v, "events": dense[v]} for v in order]

    def report(name, pred):
        mAPs, _ = compute_mAPs(truth, pred, tolerances=TOLS)
        m = [round(100 * x, 2) for x in mAPs]
        print(f"  {name:20s} mAP@0/1/2/4 = {m}  Avg {round(sum(m)/4,2)}")
        return m

    print("\n=== NMS sweep (multi-event mAP, %) ===")
    report("none (dense)", pred_dense)
    report("hard-NMS w=9", non_maximum_supression(pred_dense, 9, threshold=0.01))
    report("soft-NMS w=4", soft_non_maximum_supression(pred_dense, 4, threshold=0.01))
    report("hard-NMS w=15", non_maximum_supression(pred_dense, 15, threshold=0.01))
    print(f"\n  baseline V-JEPA->MSTCN none : {BASE['none']}  Avg {round(sum(BASE['none'])/4,2)}")
    print(f"  baseline V-JEPA->MSTCN soft : {BASE['soft']}  Avg {round(sum(BASE['soft'])/4,2)}")


if __name__ == "__main__":
    main()
