"""Does TEST-TIME input length matter for the asymmetric model? Take r12_in96_out12
(trained: input +-96 context, loss/emission central +-12) and evaluate with SHORTER
input windows at test time (emission always the central +-12). If short test input
holds up -> 'train long, test short' works; if it drops -> the context is used at
INFERENCE, not a training-only crutch."""
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

dev = "cuda"; types = ("touch", "untouch"); E = 12
en.TOLS = [0, 1, 2]
tq = torch.load(os.path.join(CACHE, "type_queries.pt"), weights_only=False)
qv = {t: tq[t].to(dev) for t in types}
fc = {}


def feats(k):
    if k not in fc:
        fc[k] = np.load(os.path.join(F3, k + ".npy")).astype(np.float32)
    return fc[k]


def main():
    model = Refine4(768, 128, 5, 3, 2, "film").to(dev)
    model.load_state_dict(torch.load(os.path.join(ROOT, "outputs/spot/r12_in96_out12/best.pt"),
                                     weights_only=False)["model"]); model.eval()
    split = "test"
    coarse = torch.load(os.path.join(CACHE, f"idx_multi_{split}_0.pt"), weights_only=False)["coarse"]
    vids = [v for v in json.load(open(os.path.join(LAB, f"{split}.json")))
            if os.path.exists(os.path.join(F3, v["video"].replace("/", "__") + ".npy"))]
    truth = [{"video": v["video"], "num_frames": v["num_frames"],
              "events": [{"label": e["label"], "frame": e["frame"]} for e in v["events"]]} for v in vids]

    print(f"model trained with input ±96 / output ±12 — test-time input sweep (emit ±{E}):")
    for Wt in (12, 24, 48, 96):
        L = 2 * Wt + 1
        det = {v["video"]: [] for v in vids}
        with torch.no_grad():
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
                            fr = c - Wt + j
                            if 0 <= fr < N:
                                x[i, j] = torch.from_numpy(feats(k)[fr]); mm[i, j] = 1
                    x = x.to(dev); mm1 = mm.unsqueeze(1).to(dev)
                    q = torch.stack([qv[t]] * len(centers))
                    ev = F.softmax(model(x, q, mm1)[-1], -1)[:, :, 1]
                    evm = ev.masked_fill(mm.to(dev) == 0, -1).cpu().numpy()
                    for c, row in zip(centers, evm):
                        for j in range(L):
                            if row[j] >= 0.01 and abs(j - Wt) <= E:
                                f2 = c - Wt + j
                                if 0 <= f2 < N:
                                    det[v["video"]].append({"label": t, "frame": int(f2), "score": float(row[j])})
        dl = [{"video": v["video"], "events": det[v["video"]]} for v in vids]
        m = en.maps_quiet(truth, en.soft_nms(dl, 4, 0.5))
        print(f"  test-input ±{Wt:>2}: sNMS @0/1/2 = {[round(x,2) for x in m]}", flush=True)


if __name__ == "__main__":
    main()
