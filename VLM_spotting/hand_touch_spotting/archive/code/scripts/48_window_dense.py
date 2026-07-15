"""Re-run the WINDOW sweep under the CORRECTED read-out (dense field + soft-NMS).
The old 'W=48 best' verdict was measured with the broken argmax read-out; the W=8/12
checkpoints (r7_*) exist, so re-evaluate them dense -- no retraining."""
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
from common.eval import non_maximum_supression                   # noqa: E402
import eval_nms as en                                            # noqa: E402

dev = "cuda"; types = ("touch", "untouch")
en.TOLS = [0, 1, 2]
tq = torch.load(os.path.join(CACHE, "type_queries.pt"), weights_only=False)
qv = {t: tq[t].to(dev) for t in types}
fc = {}

CONFIGS = [                                                      # (label, run_dir, W)
    ("W=8",       "r7_w8",      8),
    ("W=8 +rej",  "r7_w8_rej",  8),
    ("W=12",      "r7_w12",     12),
    ("W=12 +rej", "r7_w12_rej", 12),
    ("W=48",      "r6_final",   48),
]


def feats(k):
    if k not in fc:
        fc[k] = np.load(os.path.join(F3, k + ".npy")).astype(np.float32)
    return fc[k]


@torch.no_grad()
def dense_eval(model, W, coarse, vids, truth):
    L = 2 * W + 1
    det = {v["video"]: [] for v in vids}
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
                for j in range(L):
                    if row[j] >= 0.01:
                        f2 = c - W + j
                        if 0 <= f2 < N:
                            det[v["video"]].append({"label": t, "frame": int(f2), "score": float(row[j])})
    dl = [{"video": v["video"], "events": det[v["video"]]} for v in vids]
    dense = en.maps_quiet(truth, dl)
    snms = en.maps_quiet(truth, en.soft_nms(dl, 4, 0.5))
    return dense, snms


def main():
    for split in ("val", "test"):
        coarse = torch.load(os.path.join(CACHE, f"idx_multi_{split}_0.pt"), weights_only=False)["coarse"]
        vids = [v for v in json.load(open(os.path.join(LAB, f"{split}.json")))
                if os.path.exists(os.path.join(F3, v["video"].replace("/", "__") + ".npy"))]
        truth = [{"video": v["video"], "num_frames": v["num_frames"],
                  "events": [{"label": e["label"], "frame": e["frame"]} for e in v["events"]]} for v in vids]
        print(f"\n===== {split}: window sweep under DENSE read-out =====")
        print(f"{'config':<12} | {'dense @0/1/2':<22} | {'dense+softNMS @0/1/2':<22}")
        print("-" * 64)
        for label, run, W in CONFIGS:
            ck = os.path.join(ROOT, "outputs/spot", run, "best.pt")
            if not os.path.exists(ck):
                print(f"{label:<12} | missing ckpt {run}"); continue
            model = Refine4(768, 128, 5, 3, 2, "film").to(dev)
            model.load_state_dict(torch.load(ck, weights_only=False)["model"]); model.eval()
            dense, snms = dense_eval(model, W, coarse, vids, truth)
            print(f"{label:<12} | {' '.join(f'{x:5.2f}' for x in dense):<22} | "
                  f"{' '.join(f'{x:5.2f}' for x in snms):<22}", flush=True)


if __name__ == "__main__":
    main()
