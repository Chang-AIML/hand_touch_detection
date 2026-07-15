"""DECISIVE EXPERIMENT: is @0 killed by the LOCAL WINDOW or by the LLM idx?

Take the FULL-SEQUENCE query-FiLM MS-TCN (trained on whole videos, dilate=0 -> @0~22)
and evaluate it TWO ways on the SAME videos:
  (A) plain             : all per-frame detections.
  (B) idx-GATED         : keep only detections within +-W of an LLM stage-1 idx.
Both use the SAME raw V-JEPA features and (for B) the SAME idx neighborhoods the
windowed refine saw. If (B) keeps @0 ~= (A) [~20], while the windowed-TRAINED refine
gets @0 ~5 on those same neighborhoods, then the WINDOWED TRAINING REGIME is what
kills @0 -- not the idx, not the features, not the region.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "train"))
_COMMON = "/home/chang_noroot/data2/huyanh/Workspace/hand_touch_detection"
sys.path.insert(0, _COMMON)
LAB = os.path.join(_COMMON, "data", "HOI4D-v3")
F3 = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/feat_interleave"
CACHE = os.path.join(ROOT, "outputs/action/cache")
from lang_spot import SpotHead                                   # noqa: E402
from common.score import compute_mAPs                            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "outputs/spot/ls_q_F3_d0/best.pt"))
    ap.add_argument("--gate_W", type=int, nargs="+", default=[48, 16, 8])
    args = ap.parse_args()
    dev = "cuda"; types = ("touch", "untouch"); pad = 5
    tq = torch.load(os.path.join(CACHE, "type_queries.pt"), weights_only=False)
    qvec = {t: tq[t].to(dev) for t in types}
    model = SpotHead(768, 2, num_stages=3, use_film=True).to(dev)
    model.load_state_dict(torch.load(args.ckpt, weights_only=False)["model"]); model.eval()
    fc = {}

    def feats(k):
        if k not in fc:
            fc[k] = np.load(os.path.join(F3, k + ".npy")).astype(np.float32)
        return fc[k]

    @torch.no_grad()
    def dense_dets(vids):
        """full-video inference -> per (video) list of {label,frame,score} for all frames."""
        out = {}
        for v in vids:
            k = v["video"].replace("/", "__"); f = feats(k); N = f.shape[0]
            fp = np.pad(f, ((pad, pad), (0, 0)))
            x = torch.from_numpy(fp).float().unsqueeze(0).to(dev)
            m = torch.ones(1, fp.shape[0], 1, device=dev)
            evs = []
            for t in types:
                sc = F.softmax(model(x, m, qvec[t].unsqueeze(0))[-1][0], -1).cpu().numpy()[pad:-pad]
                for i in range(N):
                    if sc[i, 1] >= 0.01:
                        evs.append({"label": t, "frame": i, "score": float(sc[i, 1])})
            out[v["video"]] = evs
        return out

    def evalulate(vids, dets):
        truth = [{"video": v["video"], "num_frames": v["num_frames"],
                  "events": [{"label": e["label"], "frame": e["frame"]} for e in v["events"]]} for v in vids]
        pred = [{"video": v["video"], "events": dets[v["video"]]} for v in vids]
        m, _ = compute_mAPs(truth, pred, tolerances=[0, 1, 2, 4])
        return [round(float(m[i]) * 100, 2) for i in range(3)]

    def gate(vids, dets, coarse, W):
        g = {}
        for v in vids:
            k = v["video"].replace("/", "__")
            idxs = {t: [p for p, _ in coarse.get((k, t), [])] for t in types}
            g[v["video"]] = [e for e in dets[v["video"]]
                             if idxs[e["label"]] and min(abs(e["frame"] - p) for p in idxs[e["label"]]) <= W]
        return g

    for split, cache_name in (("val", "idx_multi_val_0.pt"), ("test", "idx_multi_test_0.pt")):
        cpath = os.path.join(CACHE, cache_name)
        if not os.path.exists(cpath):
            print(f"[skip] no {cache_name}"); continue
        coarse = torch.load(cpath, weights_only=False)["coarse"]
        vids = [v for v in json.load(open(os.path.join(LAB, f"{split}.json")))
                if os.path.exists(os.path.join(F3, v["video"].replace("/", "__") + ".npy"))]
        dets = dense_dets(vids)
        plain = evalulate(vids, dets)
        print(f"\n===== {split} (full-seq query-FiLM MS-TCN, trained on FULL videos) =====")
        print(f"  (A) plain full-seq         @0/1/2 = {plain}")
        for W in args.gate_W:
            gd = gate(vids, dets, coarse, W)
            gm = evalulate(vids, gd)
            print(f"  (B) idx-GATED  (+-{W:>2})       @0/1/2 = {gm}")
        print(f"  --- vs WINDOWED-TRAINED refine (same idx nbhd): @0~5, @2~56 ---")


if __name__ == "__main__":
    main()
