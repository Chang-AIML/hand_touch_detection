"""Full diagnostic report (per user's spec) for a trained checkpoint:
  1. GT-containing crop eval        -> local-window MAE + hit@0/1/2/4
  2. Honest full-video sliding eval -> dense-aggregation MAE + mAP@0/1/2/4
  3. Empty-window calibration       -> avg max-cos on pos vs neg windows + gap
  4. Same-video hard negatives      -> q_touch ranks touch>untouch, q_untouch vice versa

  python scripts/12_report.py --ckpt outputs/phase1/<run>/best.pt [--n 300]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/home/chang_noroot/data2/hf_cache")
sys.path.insert(0, "/home/chang_noroot/data2/huyanh/Workspace/hand_touch_detection")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--window_secs", type=int, default=10)
    ap.add_argument("--stride_secs", type=int, default=5)
    args = ap.parse_args()
    random.seed(0)

    from train.train_loop import build
    from data.dataset import Phase1Dataset
    from eval.metrics import predict_windowed, build_truth_pred
    from common.score import compute_mAPs

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    W, ad, loc, loz = build(cfg, "cuda")
    ad.load_state_dict(ck["adaptor"]); loc.load_state_dict(ck["loc"])
    if loz.sim_head is not None and "sim_head" in ck:
        loz.sim_head.load_state_dict(ck["sim_head"])
    if loz.film is not None and "film" in ck:
        loz.film.load_state_dict(ck["film"])
    ad.eval()
    fps = cfg["fps"]; temp = cfg["temp"]; win = args.window_secs
    print(f"=== REPORT {args.ckpt} (epoch {ck.get('epoch')}, split {args.split}) ===")

    # ---------- 1. GT-containing crop eval ----------
    ds_win = Phase1Dataset(args.split, cfg["feat_dir"], crop_secs=win, p_neg=0.0)
    from torch.utils.data import DataLoader
    from data.dataset import collate_list
    errs, hits = [], {0: 0, 1: 0, 2: 0, 4: 0}; n = 0
    dl = DataLoader(ds_win, batch_size=24, shuffle=False, collate_fn=collate_list, num_workers=4)
    with torch.no_grad():
        for batch in dl:
            s_list, metas = loz.forward_batch(batch)
            for s, m in zip(s_list, metas):
                sn = s.float().cpu().numpy()
                pred = int(sn.argmax()); errs.append(abs(pred - m["gt"]))
                for t in hits: hits[t] += int(abs(pred - m["gt"]) <= t)
                n += 1
            if n >= args.n: break
    print("\n[1] GT-containing crop (local-window):")
    print(f"    MAE {np.mean(errs):.2f} | hit@0/1/2/4 "
          f"{100*hits[0]/n:.1f}/{100*hits[1]/n:.1f}/{100*hits[2]/n:.1f}/{100*hits[4]/n:.1f}%")

    # ---------- 2. Honest full-video sliding ----------
    ds_full = Phase1Dataset(args.split, cfg["feat_dir"], crop_secs=0)
    preds = predict_windowed(loz, ds_full, window_secs=win, stride_secs=args.stride_secs,
                             fps=fps, batch_size=24, max_samples=args.n)
    err2 = {"all": [], "touch": [], "untouch": []}
    for p in preds:
        e = abs(p["pred_frame"] - p["gt"]); err2["all"].append(e); err2[p["type"]].append(e)
    truth, pred = build_truth_pred(preds)
    mAPs, _ = compute_mAPs(truth, pred, tolerances=[0, 1, 2, 4])
    print("\n[2] Honest full-video (dense sliding aggregation):")
    print(f"    MAE all {np.mean(err2['all']):.2f} touch {np.mean(err2['touch']):.2f} "
          f"untouch {np.mean(err2['untouch']):.2f}")
    print(f"    mAP@0/1/2/4 = {[round(100*m,2) for m in mAPs]}")

    # ---------- 3. Empty-window calibration ----------
    samples = json.load(open(os.path.join(ROOT, "data", "annotations", f"{args.split}_samples.json")))
    samples = [s for s in samples if os.path.exists(os.path.join(cfg["feat_dir"], s["video_id"] + ".npy"))]
    pos_max, neg_max = [], []
    with torch.no_grad():
        for s in samples[: args.n]:
            feats = np.load(os.path.join(cfg["feat_dir"], s["video_id"] + ".npy")).astype(np.float32)
            N = feats.shape[0]; nsec = N // fps; gt = s["frame"]; gsec = min(gt // fps, nsec - 1)
            if nsec <= win: continue
            pos_start = max(0, min(gsec, nsec - win))
            negs = [st for st in range(0, nsec - win + 1) if not (st <= gsec < st + win)]
            for tag, st_sec, out in [("pos", pos_start, pos_max), ("neg", random.choice(negs) if negs else None, neg_max)]:
                if st_sec is None: continue
                st = st_sec * fps
                b = [{"feats": torch.from_numpy(feats[st:st+win*fps]), "question": s["question"],
                      "gt": 0, "fps": fps, "type": s["type"], "video_id": s["video_id"],
                      "num_frames": win*fps, "full_num_frames": N,
                      "anchor_start_sec": st_sec, "anchor_num_secs": win}]
                sl, _ = loz.forward_batch(b)
                out.append(float(sl[0].max()) * temp)         # max cosine
    gap = np.mean(pos_max) - np.mean(neg_max)
    print("\n[3] Empty-window calibration (max cosine):")
    print(f"    positive avg {np.mean(pos_max):.3f} | negative avg {np.mean(neg_max):.3f} "
          f"| separation gap {gap:.3f}")

    # ---------- 4. Same-video hard negatives ----------
    by_vid = {}
    for s in samples:
        by_vid.setdefault(s["video_id"], {})[s["type"]] = s
    tt, uu, tot = 0, 0, 0
    with torch.no_grad():
        for vid, d in by_vid.items():
            if "touch" not in d or "untouch" not in d: continue
            ft = np.load(os.path.join(cfg["feat_dir"], vid + ".npy")).astype(np.float32)
            N = ft.shape[0]; nsec = N // fps
            gt_t, gt_u = d["touch"]["frame"], d["untouch"]["frame"]
            if abs(gt_t - gt_u) >= win * fps: continue         # need a window containing both
            lo = max(0, min(gt_t, gt_u) - fps); ctr = (gt_t + gt_u) // 2
            st_sec = max(0, min(ctr // fps - win // 2, nsec - win)); st = st_sec * fps
            if not (st <= gt_t < st + win*fps and st <= gt_u < st + win*fps): continue
            lt, lu = gt_t - st, gt_u - st
            outs = {}
            for typ in ("touch", "untouch"):
                b = [{"feats": torch.from_numpy(ft[st:st+win*fps]), "question": d[typ]["question"],
                      "gt": 0, "fps": fps, "type": typ, "video_id": vid, "num_frames": win*fps,
                      "full_num_frames": N, "anchor_start_sec": st_sec, "anchor_num_secs": win}]
                sl, _ = loz.forward_batch(b); outs[typ] = sl[0].float().cpu().numpy()
            tt += int(outs["touch"][lt] > outs["touch"][lu])
            uu += int(outs["untouch"][lu] > outs["untouch"][lt])
            tot += 1
            if tot >= args.n: break
    if tot:
        print("\n[4] Same-video hard negatives (language conditioning):")
        print(f"    q_touch ranks touch>untouch: {100*tt/tot:.1f}% | "
              f"q_untouch ranks untouch>touch: {100*uu/tot:.1f}% (n={tot})")
    else:
        print("\n[4] Same-video hard negatives: no co-occurring pairs within a window")


if __name__ == "__main__":
    main()
