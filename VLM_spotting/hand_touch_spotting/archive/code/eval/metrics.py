"""Phase 1 evaluation — AP@tIoU {0,1,2,4} + mAP + MAE (frames), touch/untouch.

Reuses the shared scorer from hand_touch_detection/common so numbers are directly
comparable to the ASTRM / TSP / V-JEPA baseline tables (same frame-tolerance AP).
Each query (video, question, type, ordinal) yields ONE predicted frame + score;
predictions are grouped per (video, label) for the standard mAP.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np
import torch

# shared scorer
_COMMON = "/home/chang_noroot/data2/huyanh/Workspace/hand_touch_detection"
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)
from common.score import compute_mAPs  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def soft_argmax_np(s: np.ndarray) -> float:
    e = np.exp(s - s.max())
    p = e / e.sum()
    return float((p * np.arange(len(s))).sum())


@torch.no_grad()
def predict(localizer, dataset, batch_size=24, use_soft_argmax=True, max_samples=0,
            log_every=0):
    """Run the localizer over a dataset; return list of per-sample prediction dicts."""
    from torch.utils.data import DataLoader
    from data.dataset import collate_list
    localizer.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_list, num_workers=4)
    preds = []
    n = 0
    for batch in loader:
        s_list, metas = localizer.forward_batch(batch)
        for s, m in zip(s_list, metas):
            s_np = s.float().cpu().numpy()
            e = np.exp(s_np - s_np.max()); p = e / e.sum()      # softmax over frames
            pf = float((p * np.arange(len(s_np))).sum()) if use_soft_argmax else float(s_np.argmax())
            preds.append({"video_id": m["video_id"], "type": m["type"], "gt": m["gt"],
                          "pred_frame": pf, "score": float(p.max())})   # confidence in (0,1]
        n += len(batch)
        if log_every and n % log_every < batch_size:
            print(f"    [eval] {n}/{len(dataset)}", flush=True)
        if max_samples and n >= max_samples:
            break
    return preds


@torch.no_grad()
def predict_windowed(localizer, dataset, window_secs=10, stride_secs=5, fps=15,
                     batch_size=24, max_samples=0, return_curves=False):
    """Sliding-window DENSE aggregation over the full (300-frame or irregular) video.

    The model is trained on `window_secs` crops. We slide windows across the whole
    video and, for each GLOBAL frame, AVERAGE its similarity s(t) over every window
    that covers it -> one full-video score curve -> global argmax. The true event
    frame is consistently high across windows; spurious single-window peaks average
    out. Far more robust than picking one window by confidence.
    Returns per-sample prediction dicts (pred_frame in FULL-video coords)."""
    from torch.utils.data import DataLoader
    from data.dataset import collate_list
    localizer.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_list,
                        num_workers=4)
    # accumulate per (video,question): sum & count of s(t) per global frame
    acc = {}   # key -> dict(sum=np[N], cnt=np[N], gt, typ, vid, N)
    win_batch, keys = [], []

    def flush():
        for i in range(0, len(win_batch), batch_size):
            chunk = win_batch[i:i + batch_size]
            kk = keys[i:i + batch_size]
            s_list, _ = localizer.forward_batch(chunk)
            for (key, start, W), s in zip(kk, s_list):
                sn = s.float().cpu().numpy()
                # per-window confidence = peak softmax prob (flat/non-event windows ~1/W)
                e = np.exp(sn - sn.max()); p = e / e.sum()
                conf = float(p.max()) * len(sn)               # >=1; ~1 if flat, big if peaked
                a = acc[key]
                a["sum"][start:start + len(sn)] += conf * p    # weight peaked windows more
                a["cnt"][start:start + len(sn)] += conf
        win_batch.clear(); keys.clear()

    n = 0
    for batch in loader:
        s = batch[0]
        feats = s["feats"]; N = feats.shape[0]; gt_g = s["gt"]
        num_secs = N // fps
        W = window_secs * fps
        key = (s["video_id"], s["question"])
        acc[key] = {"sum": np.zeros(N), "cnt": np.zeros(N) + 1e-9,
                    "gt": gt_g, "typ": s["type"], "vid": s["video_id"], "N": N}
        starts = list(range(0, max(1, num_secs - window_secs + 1), stride_secs))
        if num_secs > window_secs and starts[-1] != num_secs - window_secs:
            starts.append(num_secs - window_secs)
        for st_sec in starts:
            st = st_sec * fps
            win_batch.append({
                "feats": feats[st:st + W], "question": s["question"],
                "gt": max(0, min(W - 1, gt_g - st)), "fps": fps, "type": s["type"],
                "video_id": s["video_id"], "num_frames": min(W, N - st),
                "full_num_frames": N, "anchor_start_sec": st_sec,
                "anchor_num_secs": window_secs})
            keys.append((key, st, W))
        n += 1
        if len(win_batch) >= 256:
            flush()
        if max_samples and n >= max_samples:
            break
    flush()
    preds = []
    for key, a in acc.items():
        curve = a["sum"] / a["cnt"]                       # (N,) mean similarity per frame
        pf = int(curve.argmax())
        c = curve - curve.max(); pmax = float(np.exp(0) / np.exp(c).sum())
        d = {"video_id": a["vid"], "type": a["typ"], "gt": a["gt"],
             "pred_frame": float(pf), "score": pmax}
        if return_curves:
            d["curve"] = curve
        preds.append(d)
    return preds


def nms_peaks(curve, k=8, min_gap=15, rel_thresh=0.5):
    """Pick up to k peaks of a full-video score curve, >= rel_thresh*max, min_gap apart
    (greedy NMS). Returns list of (frame, score)."""
    import numpy as np
    top = float(curve.max()); out = []
    for i in np.argsort(-curve):
        if curve[i] < rel_thresh * top:
            break
        if all(abs(int(i) - p) >= min_gap for p, _ in out):
            out.append((int(i), float(curve[i])))
        if len(out) >= k:
            break
    return out


@torch.no_grad()
def evaluate_multievent(localizer, split, feat_dir, fps=15, window_secs=10,
                        stride_secs=5, tolerances=(0, 1, 2, 4), batch_size=24,
                        min_gap=15, rel_thresh=0.5, max_videos=0,
                        question_phase="canonical", categories=None,
                        exclude_categories=None, types=("touch", "untouch")):
    """Multi-event detection eval. question_phase: 'canonical' (default GENERIC_Q) |
    'test' (random HELD-OUT phrasing, for the language test). categories /
    exclude_categories filter videos (for seen/unseen generalization)."""
    import os, json, random
    import numpy as np
    from data.dataset import GENERIC_Q, ROOT, video_category
    from data.questions import sample_question
    _COMMON = "/home/chang_noroot/data2/huyanh/Workspace/hand_touch_detection"
    if _COMMON not in sys.path:
        sys.path.insert(0, _COMMON)
    from common.score import compute_mAPs

    vids = json.load(open(os.path.join(ROOT, "data", "annotations", f"{split}.json")))
    vids = [v for v in vids if os.path.exists(os.path.join(feat_dir, v["video_id"] + ".npy"))]
    if categories is not None:
        vids = [v for v in vids if video_category(v["video_id"]) in categories]
    if exclude_categories is not None:
        vids = [v for v in vids if video_category(v["video_id"]) not in exclude_categories]
    if max_videos:
        vids = vids[:max_videos]
    rng = random.Random(0)

    def q_for(typ):
        return GENERIC_Q[typ] if question_phase == "canonical" else sample_question(typ, question_phase, rng)

    class _DS:
        def __init__(self, items): self.items = items
        def __len__(self): return len(self.items)
        def __getitem__(self, i):
            v, typ = self.items[i]
            f = np.load(os.path.join(feat_dir, v + ".npy")).astype(np.float32)
            return {"feats": torch.from_numpy(f), "question": q_for(typ), "gt": 0,
                    "fps": fps, "type": typ, "video_id": v, "num_frames": f.shape[0],
                    "full_num_frames": f.shape[0], "anchor_start_sec": 0,
                    "anchor_num_secs": f.shape[0] // fps}
    items = [(v["video_id"], typ) for v in vids for typ in types
             if any(e["type"] == typ for e in v["events"])]
    preds = predict_windowed(localizer, _DS(items), window_secs=window_secs,
                             stride_secs=stride_secs, fps=fps, batch_size=batch_size,
                             return_curves=True)
    tr = {v["video_id"]: [{"label": e["type"], "frame": e["frame"]} for e in v["events"]] for v in vids}
    pr = {v["video_id"]: [] for v in vids}
    for p in preds:
        for fr, sc in nms_peaks(p["curve"], min_gap=min_gap, rel_thresh=rel_thresh):
            pr[p["video_id"]].append({"label": p["type"], "frame": fr, "score": sc})
    order = sorted(pr)
    truth = [{"video": v, "events": tr[v]} for v in order]
    pred = [{"video": v, "events": pr[v]} for v in order]
    mAPs, _ = compute_mAPs(truth, pred, tolerances=list(tolerances))
    ap = {f"mAP@{t}": float(mAPs[i]) * 100 for i, t in enumerate(tolerances)}
    ap["mAP_012"] = float(np.mean([mAPs[i] for i, t in enumerate(tolerances) if t in (0, 1, 2)])) * 100
    return ap


def build_truth_pred(preds):
    """Group per-sample preds into the compute_mAPs (truth, pred) structures."""
    pred_by_vid = defaultdict(list)
    truth_by_vid = defaultdict(list)
    for p in preds:
        pred_by_vid[p["video_id"]].append(
            {"label": p["type"], "frame": int(round(p["pred_frame"])), "score": p["score"]})
        truth_by_vid[p["video_id"]].append({"label": p["type"], "frame": int(p["gt"])})
    vids = sorted(pred_by_vid)
    pred = [{"video": v, "events": pred_by_vid[v]} for v in vids]
    truth = [{"video": v, "events": truth_by_vid[v]} for v in vids]
    return truth, pred


def evaluate(localizer, dataset, tolerances=(0, 1, 2, 4), batch_size=24,
             use_soft_argmax=True):
    preds = predict(localizer, dataset, batch_size=batch_size,
                    use_soft_argmax=use_soft_argmax)
    # MAE (frames), overall + per type
    err = defaultdict(list)
    for p in preds:
        e = abs(p["pred_frame"] - p["gt"])
        err["all"].append(e)
        err[p["type"]].append(e)
    mae = {k: float(np.mean(v)) for k, v in err.items()}

    truth, pred = build_truth_pred(preds)
    mAPs, _tols = compute_mAPs(truth, pred, tolerances=list(tolerances))   # mAPs in [0,1]
    ap = {f"mAP@{t}": float(mAPs[i]) * 100 for i, t in enumerate(tolerances)}
    ap["mAP_012"] = float(np.mean([mAPs[i] for i, t in enumerate(tolerances)
                                   if t in (0, 1, 2)])) * 100
    return {"mae": mae, "ap": ap, "n": len(preds)}
