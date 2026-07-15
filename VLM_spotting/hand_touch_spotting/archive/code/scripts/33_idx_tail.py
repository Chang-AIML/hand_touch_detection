"""Characterize the long-tail localization errors of an idx checkpoint: real
(unclipped) error stats, worst cases, and WHY they fail (count mismatch vs
positional, event ordinal, position-in-video)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/home/chang_noroot/data2/hf_cache")
FEAT = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/feat_interleave"
CK = os.path.join(ROOT, "outputs/idx/idx_multi/best.pt")


def main():
    from models.wrapper import QwenWrapper
    from models.vjepa_adaptor import VJEPAAdaptor
    from models.idx_localizer import IdxLocalizer
    from data.questions import GENERIC_Q

    ck = torch.load(CK, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]; rank = cfg.get("lora_rank", 16)
    W = QwenWrapper(device="cuda", dtype=torch.bfloat16)
    ad = VJEPAAdaptor(768, W.d_llm, hidden=2048, n_layers=2).to("cuda", torch.bfloat16)
    ad.set_target_rms_from(W.embed_tokens.weight)
    ad.load_state_dict(ck["adaptor"]); ad.eval()
    if cfg.get("lora"):
        lp = W.add_lora(rank=rank, alpha=2 * rank, n_layers=W.n_layers, target="all")
        for p, s in zip(lp, ck["lora"]):
            p.data.copy_(s.to(p.device, p.dtype))
    loz = IdxLocalizer(W, ad, use_idx=True, use_anchor=True, anchor_stride=5,
                       anchor_max_side=252, use_mrope=False, fps=15, max_frames=320,
                       grad_checkpoint=False)

    vids = json.load(open(os.path.join(ROOT, "data", "annotations", "val.json")))
    vids = [v for v in vids if os.path.exists(os.path.join(FEAT, v["video_id"] + ".npy"))]
    types = ("touch", "untouch")
    items = [(v["video_id"], t) for v in vids for t in types
             if any(e["type"] == t for e in v["events"])]
    fc = {}

    def feats(vid):
        if vid not in fc:
            fc[vid] = np.load(os.path.join(FEAT, vid + ".npy")).astype(np.float32)
        return fc[vid]

    gt_by, pred_by = {}, {}
    for b0 in range(0, len(items), 8):
        chunk = items[b0:b0 + 8]
        samples = [{"feats": torch.from_numpy(feats(vid)), "question": GENERIC_Q[t],
                    "event_frames": None, "gt": -1, "fps": 15, "type": t, "video_id": vid,
                    "num_frames": feats(vid).shape[0], "full_num_frames": feats(vid).shape[0],
                    "anchor_start_sec": 0, "anchor_num_secs": feats(vid).shape[0] // 15}
                   for vid, t in chunk]
        outs = loz.predict_multievent_batch(samples)
        for (vid, t), (fr, sc) in zip(chunk, outs):
            N = feats(vid).shape[0]
            pred_by[(vid, t)] = sorted(int(f) for f in fr if 0 <= f < N)

    for v in vids:
        for t in types:
            g = sorted(e["frame"] for e in v["events"] if e["type"] == t)
            if g:
                gt_by[(v["video_id"], t)] = g

    # per-GT-event error + metadata
    rows = []   # (err, vid, type, ordinal, n_gt, n_pred, gt_frame, nearest_pred, N)
    for (vid, t), gts in gt_by.items():
        preds = pred_by.get((vid, t), [])
        N = feats(vid).shape[0]
        for oi, gf in enumerate(gts):
            e = min((abs(gf - p) for p in preds), default=10 ** 9)
            rows.append((e, vid, t, oi + 1, len(gts), len(preds), gf, N))
    errs = np.array([r[0] for r in rows])
    print(f"n events {len(errs)} | max err {errs.max()} | "
          f"p50 {np.percentile(errs,50):.0f} p90 {np.percentile(errs,90):.0f} "
          f"p95 {np.percentile(errs,95):.0f} p99 {np.percentile(errs,99):.0f}")
    for lo, hi in [(0, 4), (4, 8), (8, 16), (16, 30), (30, 60), (60, 120), (120, 10 ** 9)]:
        m = (errs >= lo) & (errs < hi)
        print(f"  err [{lo:>3},{hi if hi < 1000 else 'inf':>3}): {m.sum():>3}  ({100*m.mean():.1f}%)")

    tail = [r for r in rows if r[0] >= 16]
    print(f"\n=== TAIL (err>=16): {len(tail)} events ===")
    # count-mismatch: does the video have fewer preds than gts for that type?
    cm = sum(1 for r in tail if r[5] < r[4])
    print(f"  of tail, {cm} ({100*cm/max(len(tail),1):.0f}%) are in (video,type) where n_pred < n_gt (MISSED count)")
    # ordinal / position breakdown
    from collections import Counter
    ordc = Counter(r[3] for r in tail); posc = Counter()
    for r in tail:
        gf, N = r[6], r[7]
        posc["early<20%" if gf < 0.2 * N else ("late>80%" if gf > 0.8 * N else "mid")] += 1
    print(f"  by ordinal (which occurrence): {dict(sorted(ordc.items()))}")
    print(f"  by position-in-video: {dict(posc)}")
    print(f"  by type: {dict(Counter(r[2] for r in tail))}")
    print("\n  worst 12 (err, type, ord/n_gt, gt_frame, n_pred, video):")
    for r in sorted(tail, reverse=True)[:12]:
        e, vid, t, o, ng, npr, gf, N = r
        print(f"    err {e:>3} | {t:<7} ev{o}/{ng} gt={gf:<3} npred={npr} | {vid} | preds={pred_by.get((vid,t))}")


if __name__ == "__main__":
    main()
