"""Decouple INPUT CONTEXT from EMISSION RESPONSIBILITY: take models trained with wide
windows (context W_train = 48/64/96, multi-GT labels) and at eval emit ONLY the central
+-E frames around each LLM idx. If W_train=96 + E=12 beats the W=12-TRAINED model (61.7),
the small-window bottleneck is CONTEXT, not the emission region -> design = wide sight,
narrow responsibility (LLM localization stays load-bearing: emission covers ~35%)."""
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

dev = "cuda"; types = ("touch", "untouch")
en.TOLS = [0, 1, 2]
tq = torch.load(os.path.join(CACHE, "type_queries.pt"), weights_only=False)
qv = {t: tq[t].to(dev) for t in types}
fc = {}

MODELS = [("r10_w48_mg", 48), ("r10_w64_mg", 64), ("r10_w96_mg", 96)]
EMITS = [8, 12, 24, None]                                        # None = full window


def feats(k):
    if k not in fc:
        fc[k] = np.load(os.path.join(F3, k + ".npy")).astype(np.float32)
    return fc[k]


@torch.no_grad()
def fields(model, W, coarse, vids):
    """per video: list of (center, type, prob-row) for each idx window."""
    L = 2 * W + 1; out = {v["video"]: [] for v in vids}
    for v in vids:
        k = v["video"].replace("/", "__"); N = feats(k).shape[0]
        for t in types:
            dets = coarse.get((k, t), [])
            if not dets:
                continue
            centers = [int(p) for p, _ in dets]
            x = torch.zeros(len(centers), L, 768); mm = torch.zeros(len(centers), L)
            for i, c in enumerate(centers):
                for j in range(L):
                    fr = c - W + j
                    if 0 <= fr < N:
                        x[i, j] = torch.from_numpy(feats(k)[fr]); mm[i, j] = 1
            x = x.to(dev); mm1 = mm.unsqueeze(1).to(dev)
            q = torch.stack([qv[t]] * len(centers))
            ev = F.softmax(model(x, q, mm1)[-1], -1)[:, :, 1]
            evm = ev.masked_fill(mm.to(dev) == 0, -1).cpu().numpy()
            for c, row in zip(centers, evm):
                out[v["video"]].append((c, t, row, N))
    return out


def emit(flds, vids, W, E):
    det = {v["video"]: [] for v in vids}
    for v in vids:
        for c, t, row, N in flds[v["video"]]:
            for j in range(len(row)):
                if row[j] < 0.01:
                    continue
                if E is not None and abs(j - W) > E:
                    continue
                f2 = c - W + j
                if 0 <= f2 < N:
                    det[v["video"]].append({"label": t, "frame": int(f2), "score": float(row[j])})
    return det


def main():
    for split in ("test", "val"):
        coarse = torch.load(os.path.join(CACHE, f"idx_multi_{split}_0.pt"), weights_only=False)["coarse"]
        vids = [v for v in json.load(open(os.path.join(LAB, f"{split}.json")))
                if os.path.exists(os.path.join(F3, v["video"].replace("/", "__") + ".npy"))]
        truth = [{"video": v["video"], "num_frames": v["num_frames"],
                  "events": [{"label": e["label"], "frame": e["frame"]} for e in v["events"]]} for v in vids]
        print(f"\n===== {split}: context(W_train) x emission(E), soft-NMS @0/1/2 =====")
        print(f"{'model':<14} " + " ".join(f"{'E=%s'%(e if e else 'full'):<20}" for e in EMITS))
        for run, W in MODELS:
            model = Refine4(768, 128, 5, 3, 2, "film").to(dev)
            model.load_state_dict(torch.load(os.path.join(ROOT, "outputs/spot", run, "best.pt"),
                                             weights_only=False)["model"]); model.eval()
            flds = fields(model, W, coarse, vids)
            cells = []
            for E in EMITS:
                det = emit(flds, vids, W, E)
                dl = [{"video": v["video"], "events": det[v["video"]]} for v in vids]
                m = en.maps_quiet(truth, en.soft_nms(dl, 4, 0.5))
                cells.append("/".join(f"{x:.1f}" for x in m))
            print(f"{run:<14} " + " ".join(f"{c:<20}" for c in cells), flush=True)
        print("reference W=12-trained, E=12: test 10.5/42.7/61.7 | W=8-trained small: 9.3/41.0/59.5")


if __name__ == "__main__":
    main()
