"""Test the read-out hypothesis: refine4's low mAP@0 is because it emits ONE hard-argmax
detection per idx (no score-ranking of exact vs near-miss). Emit the DENSE scored field
from each window instead (like full-seq) and see if mAP@0 jumps -- no retraining."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "train"))
_COMMON = "/home/chang_noroot/data2/huyanh/Workspace/hand_touch_detection"
sys.path.insert(0, _COMMON); sys.path.insert(0, os.path.join(_COMMON, "methods/spot_head"))
LAB = os.path.join(_COMMON, "data", "HOI4D-v3")
F3 = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/feat_interleave"
CACHE = os.path.join(ROOT, "outputs/action/cache")
from refine4 import Refine4                                      # noqa: E402
import eval_nms as en                                            # noqa: E402

dev = "cuda"; types = ("touch", "untouch"); W = 48
en.TOLS = [0, 1, 2]
tq = torch.load(os.path.join(CACHE, "type_queries.pt"), weights_only=False)
qv = {t: tq[t].to(dev) for t in types}
fc = {}


def feats(k):
    if k not in fc:
        fc[k] = np.load(os.path.join(F3, k + ".npy")).astype(np.float32)
    return fc[k]


def main(split="val"):
    va = torch.load(os.path.join(CACHE, f"idx_multi_{split}_0.pt"), weights_only=False)["coarse"]
    vids = [v for v in json.load(open(os.path.join(LAB, f"{split}.json")))
            if os.path.exists(os.path.join(F3, v["video"].replace("/", "__") + ".npy"))]
    truth = [{"video": v["video"], "num_frames": v["num_frames"],
              "events": [{"label": e["label"], "frame": e["frame"]} for e in v["events"]]} for v in vids]
    model = Refine4(768, 128, 5, 3, 2, "film").to(dev)
    model.load_state_dict(torch.load(os.path.join(ROOT, "outputs/spot/r6_final/best.pt"),
                                     weights_only=False)["model"]); model.eval()

    # per (video,type): collect window event-prob fields
    det_argmax = {v["video"]: [] for v in vids}
    det_dense = {v["video"]: [] for v in vids}
    det_top3 = {v["video"]: [] for v in vids}

    @torch.no_grad()
    def run():
        for v in vids:
            k = v["video"].replace("/", "__"); N = feats(k).shape[0]
            for t in types:
                dets = va.get((k, t), [])
                if not dets:
                    continue
                centers = [int(p) for p, _ in dets]; L = 2 * W + 1
                x = torch.zeros(len(centers), L, 768); mm = torch.zeros(len(centers), L)
                for i, c in enumerate(centers):
                    for j in range(L):
                        fr = c - W + j
                        if 0 <= fr < N:
                            x[i, j] = torch.from_numpy(feats(k)[fr]); mm[i, j] = 1
                x = x.to(dev); mm1 = mm.unsqueeze(1).to(dev)
                q = torch.stack([qv[t]] * len(centers))
                ev = F.softmax(model(x, q, mm1)[-1], -1)[:, :, 1]           # [B,L]
                evm = ev.masked_fill(mm.to(dev) == 0, -1).cpu().numpy()
                for c, row in zip(centers, evm):
                    # argmax (current)
                    a = int(np.argmax(row)); fr = int(min(max(c - W + a, 0), N - 1))
                    det_argmax[v["video"]].append({"label": t, "frame": fr, "score": float(row[a])})
                    # dense: all valid frames with their prob as score
                    for j in range(L):
                        if row[j] >= 0.01:
                            f2 = c - W + j
                            if 0 <= f2 < N:
                                det_dense[v["video"]].append({"label": t, "frame": int(f2), "score": float(row[j])})
                    # top-3 highest-prob frames in the window
                    for j in np.argsort(-row)[:3]:
                        if row[j] >= 0.01:
                            f2 = c - W + int(j)
                            if 0 <= f2 < N:
                                det_top3[v["video"]].append({"label": t, "frame": int(f2), "score": float(row[j])})
    run()

    def score(det, name):
        pl = [{"video": v["video"], "events": det[v["video"]]} for v in vids]
        m = en.maps_quiet(truth, pl)
        print(f"  {name:<28} @0/1/2 = {[round(x,2) for x in m]}")

    from common.eval import non_maximum_supression
    print(f"\n=== refine4 (r6_final) {split} — read-out variants ===")
    score(det_argmax, "1 argmax/idx (current)")
    score(det_dense, "DENSE field/window")
    score(det_top3, "top-3/window")
    dl = [{"video": v["video"], "events": det_dense[v["video"]]} for v in vids]
    for w in (1, 2, 3):
        pld = non_maximum_supression(dl, w)
        print(f"  {'DENSE + NMS(w=%d)'%w:<28} @0/1/2 = {[round(x,2) for x in en.maps_quiet(truth, pld)]}")
    for w, s in ((4, 0.5), (2, 0.5), (2, 0.3)):
        pls = en.soft_nms(dl, w, s)
        print(f"  {'DENSE + softNMS(w=%d,s=%.1f)'%(w,s):<28} @0/1/2 = {[round(x,2) for x in en.maps_quiet(truth, pls)]}")


if __name__ == "__main__":
    main("val")
    main("test")
