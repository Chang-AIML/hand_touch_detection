"""LONG-VIDEO diagnostic: does MS-TCN degrade as the video gets longer?
Concatenate K short HOI4D videos (V-JEPA features) into one long sequence; events
land at offset frame positions. Run the full-seq multiclass MS-TCN (ls_mc_F3_d0) on
each length K in {1,2,3,5,10} and report mAP@0/1/2. MS-TCN is fully convolutional, so
if mAP is ~flat vs K, it is length-ROBUST -> 'long video favors LLM' fails for plain
detection, and the LLM's edge must come from GLOBAL queries (first/last/Nth), not length.
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "train"))
_COMMON = "/home/chang_noroot/data2/huyanh/Workspace/hand_touch_detection"
sys.path.insert(0, _COMMON)
LAB = os.path.join(_COMMON, "data", "HOI4D-v3")
F3 = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/feat_interleave"
from lang_spot import SpotHead                                   # noqa: E402
from common.score import compute_mAPs                            # noqa: E402


def main():
    dev = "cuda"; pad = 5
    model = SpotHead(768, 3, num_stages=3, use_film=False).to(dev)
    model.load_state_dict(torch.load(os.path.join(ROOT, "outputs/spot/ls_mc_F3_d0/best.pt"),
                                     weights_only=False)["model"]); model.eval()
    vids = [v for v in json.load(open(os.path.join(LAB, "val.json")))
            if os.path.exists(os.path.join(F3, v["video"].replace("/", "__") + ".npy"))]
    fc = {}

    def feats(v):
        k = v["video"].replace("/", "__")
        if k not in fc:
            fc[k] = np.load(os.path.join(F3, k + ".npy")).astype(np.float32)
        return fc[k]

    @torch.no_grad()
    def infer(feat):                                             # feat[N,768] -> per-frame 3-class probs
        fp = np.pad(feat, ((pad, pad), (0, 0)))
        x = torch.from_numpy(fp).float().unsqueeze(0).to(dev)
        m = torch.ones(1, fp.shape[0], 1, device=dev)
        return F.softmax(model(x, m)[-1][0], -1).cpu().numpy()[pad:-pad]

    print(f"{'K':>3} {'nseq':>5} {'frames':>7} {'events':>7} | {'mAP@0':>6} {'mAP@1':>6} {'mAP@2':>6}")
    print("-" * 56)
    for K in (1, 2, 3, 5, 10):
        rng = random.Random(0); order = vids[:]; rng.shuffle(order)
        groups = [order[i:i + K] for i in range(0, len(order) - K + 1, K)]
        truth, pred = [], []
        for gi, grp in enumerate(groups):
            feat = np.concatenate([feats(v) for v in grp], 0)
            off, evs = 0, []
            for v in grp:
                for e in v["events"]:
                    evs.append({"label": e["label"], "frame": e["frame"] + off})
                off += v["num_frames"]
            sc = infer(feat); N = feat.shape[0]
            det = [{"label": lab, "frame": i, "score": float(sc[i, c])}
                   for i in range(N) for c, lab in ((1, "touch"), (2, "untouch")) if sc[i, c] >= 0.01]
            truth.append({"video": f"long{K}_{gi}", "num_frames": N, "events": evs})
            pred.append({"video": f"long{K}_{gi}", "events": det})
        m, _ = compute_mAPs(truth, pred, tolerances=[0, 1, 2])
        tot_e = sum(len(t["events"]) for t in truth); tot_f = sum(t["num_frames"] for t in truth)
        print(f"{K:>3} {len(groups):>5} {tot_f//len(groups):>7} {tot_e/len(groups):>7.1f} | "
              f"{m[0]*100:>6.2f} {m[1]*100:>6.2f} {m[2]*100:>6.2f}", flush=True)


if __name__ == "__main__":
    main()
