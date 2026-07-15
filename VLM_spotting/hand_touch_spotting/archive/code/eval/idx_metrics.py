"""Full-video MULTI-EVENT mAP for the idx-decode model — directly comparable to the
baseline tables (cosine 41 / MSTCN 68). One forward per (video, type): the LLM reads
the whole 300-frame video and GENERATES the list of event frame indices; each detected
frame gets a score = mean digit-token probability (generation likelihood). Detections
are scored against ALL GT events with the shared frame-tolerance mAP.
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_COMMON = "/home/chang_noroot/data2/huyanh/Workspace/hand_touch_detection"
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)
from common.score import compute_mAPs                # noqa: E402


def _video_category(video_id):
    import re
    m = re.match(r"H\d+_C(\d+)_", video_id)
    return int(m.group(1)) if m else -1


@torch.no_grad()
def evaluate_multievent_idx(loz, split, feat_dir, fps=15, tolerances=(0, 1, 2, 4),
                            batch_size=8, question_phase="canonical", categories=None,
                            exclude_categories=None, types=("touch", "untouch"),
                            max_videos=0, log_every=0):
    from data.questions import GENERIC_Q, sample_question
    loz.W.model.eval(); loz.ad.eval()
    rng = random.Random(0)

    vids = json.load(open(os.path.join(ROOT, "data", "annotations", f"{split}.json")))
    vids = [v for v in vids if os.path.exists(os.path.join(feat_dir, v["video_id"] + ".npy"))]
    if categories is not None:
        vids = [v for v in vids if _video_category(v["video_id"]) in categories]
    if exclude_categories is not None:
        vids = [v for v in vids if _video_category(v["video_id"]) not in exclude_categories]
    if max_videos:
        vids = vids[:max_videos]

    def q_for(typ):
        return GENERIC_Q[typ] if question_phase == "canonical" else sample_question(typ, question_phase, rng)

    # feat cache per video (loaded once)
    feat_cache = {}

    def feats_of(vid):
        if vid not in feat_cache:
            feat_cache[vid] = np.load(os.path.join(feat_dir, vid + ".npy")).astype(np.float32)
        return feat_cache[vid]

    items = [(v["video_id"], typ) for v in vids for typ in types
             if any(e["type"] == typ for e in v["events"])]
    truth = {v["video_id"]: [{"label": e["type"], "frame": e["frame"]}
                             for e in v["events"] if e["type"] in types] for v in vids}
    pr = {v["video_id"]: [] for v in vids}

    for b0 in range(0, len(items), batch_size):
        chunk = items[b0:b0 + batch_size]
        samples = []
        for vid, typ in chunk:
            f = feats_of(vid); N = f.shape[0]
            samples.append({"feats": torch.from_numpy(f), "question": q_for(typ),
                            "event_frames": None, "gt": -1, "fps": fps, "type": typ,
                            "video_id": vid, "num_frames": N, "full_num_frames": N,
                            "anchor_start_sec": 0, "anchor_num_secs": N // fps})
        outs = loz.predict_multievent_batch(samples)
        for (vid, typ), (frames, scores) in zip(chunk, outs):
            N = feats_of(vid).shape[0]
            for fr, sc in zip(frames, scores):
                if 0 <= fr < N:
                    pr[vid].append({"label": typ, "frame": int(fr), "score": float(sc)})
        if log_every and (b0 // batch_size) % log_every == 0:
            print(f"    [eval] {b0}/{len(items)}", flush=True)

    order = sorted(truth)
    truth_l = [{"video": v, "events": truth[v]} for v in order]
    pred_l = [{"video": v, "events": pr[v]} for v in order]
    mAPs, _ = compute_mAPs(truth_l, pred_l, tolerances=list(tolerances))
    ap = {f"mAP@{t}": float(mAPs[i]) * 100 for i, t in enumerate(tolerances)}
    ap["mAP_012"] = float(np.mean([mAPs[i] for i, t in enumerate(tolerances) if t in (0, 1, 2)])) * 100
    # diagnostics: avg #detections vs #GT
    ndet = np.mean([len(pr[v]) for v in order]) if order else 0
    ngt = np.mean([len(truth[v]) for v in order]) if order else 0
    ap["_avg_ndet"] = float(ndet); ap["_avg_ngt"] = float(ngt)
    return ap
