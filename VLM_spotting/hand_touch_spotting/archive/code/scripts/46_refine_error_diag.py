"""Diagnose refine4's @0: signed-error mean (off-by-one?) + error histogram + exact-hit
fraction, for (a) stage-1 idx, (b) refine4 refined preds, (c) full-seq MS-TCN read-out
on the SAME GT events. If refine spreads to +-1~2 with low mass at 0 while full-seq
spikes at 0 -> confirms it is the windowed-training regime, not a read-out/offset bug.
"""
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
sys.path.insert(0, _COMMON)
LAB = os.path.join(_COMMON, "data", "HOI4D-v3")
F3 = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/feat_interleave"
CACHE = os.path.join(ROOT, "outputs/action/cache")
from refine4 import Refine4                                      # noqa: E402
from lang_spot import SpotHead                                   # noqa: E402

dev = "cuda"; types = ("touch", "untouch"); W = 48; pad = 5
tq = torch.load(os.path.join(CACHE, "type_queries.pt"), weights_only=False)
qv = {t: tq[t].to(dev) for t in types}
fc = {}


def feats(k):
    if k not in fc:
        fc[k] = np.load(os.path.join(F3, k + ".npy")).astype(np.float32)
    return fc[k]


def stats(name, errs):
    e = np.array(errs)
    print(f"\n{name}  (n={len(e)})")
    print(f"  mean signed err = {e.mean():+.3f}   median = {np.median(e):+.1f}   std = {e.std():.2f}")
    print(f"  EXACT (err==0): {100*np.mean(e==0):5.1f}%   |err|<=1: {100*np.mean(np.abs(e)<=1):5.1f}%   "
          f"|err|<=2: {100*np.mean(np.abs(e)<=2):5.1f}%")
    hist = {d: int(np.sum(e == d)) for d in range(-5, 6)}
    bar = "  ".join(f"{d:+d}:{hist[d]}" for d in range(-5, 6))
    print(f"  hist[-5..+5]: {bar}   (|err|>5: {int(np.sum(np.abs(e) > 5))})")


def main():
    va = torch.load(os.path.join(CACHE, "idx_multi_val_0.pt"), weights_only=False)["coarse"]
    vids = [v for v in json.load(open(os.path.join(LAB, "val.json")))
            if os.path.exists(os.path.join(F3, v["video"].replace("/", "__") + ".npy"))]

    refm = Refine4(768, 128, 5, 3, 2, "film").to(dev)
    refm.load_state_dict(torch.load(os.path.join(ROOT, "outputs/spot/r6_final/best.pt"),
                                    weights_only=False)["model"]); refm.eval()
    fsm = SpotHead(768, 3, num_stages=3, use_film=False).to(dev)
    fsm.load_state_dict(torch.load(os.path.join(ROOT, "outputs/spot/ls_mc_F3_d0/best.pt"),
                                   weights_only=False)["model"]); fsm.eval()

    err_s1, err_ref, err_fs = [], [], []
    cls = {"touch": 1, "untouch": 2}

    @torch.no_grad()
    def fs_probs(k):
        f = feats(k); fp = np.pad(f, ((pad, pad), (0, 0)))
        x = torch.from_numpy(fp).float().unsqueeze(0).to(dev)
        m = torch.ones(1, fp.shape[0], 1, device=dev)
        return F.softmax(fsm(x, m)[-1][0], -1).cpu().numpy()[pad:-pad]     # [N,3]

    for v in vids:
        k = v["video"].replace("/", "__"); N = feats(k).shape[0]
        fsp = fs_probs(k)
        for t in types:
            gts = sorted(e["frame"] for e in v["events"] if e["label"] == t)
            if not gts:
                continue
            # (c) full-seq read-out: for each GT, best peak within +-W of GT
            for g in gts:
                lo, hi = max(0, g - W), min(N, g + W + 1)
                pk = lo + int(np.argmax(fsp[lo:hi, cls[t]]))
                err_fs.append(pk - g)
            # (a) stage-1 idx  &  (b) refine, over the stage-1 detections
            dets = va.get((k, t), [])
            if not dets:
                continue
            centers = [int(p) for p, _ in dets]
            L = 2 * W + 1
            x = torch.zeros(len(centers), L, 768); mm = torch.zeros(len(centers), L)
            for i, c in enumerate(centers):
                for j in range(L):
                    fr = c - W + j
                    if 0 <= fr < N:
                        x[i, j] = torch.from_numpy(feats(k)[fr]); mm[i, j] = 1
            x = x.to(dev); mm1 = mm.unsqueeze(1).to(dev)
            q = torch.stack([qv[t]] * len(centers))
            ev = F.softmax(refm(x, q, mm1)[-1], -1)[:, :, 1].masked_fill(mm.to(dev) == 0, -1)
            off = (ev.argmax(-1) - W).cpu().numpy()
            for c, o in zip(centers, off):
                g = min(gts, key=lambda z: abs(z - c))
                if abs(g - c) <= W:                              # refinable case
                    err_s1.append(c - g)
                    err_ref.append(int(min(max(c + o, 0), N - 1)) - g)

    print("=" * 70)
    print("HOI4D val — signed error (pred - GT) on REFINABLE events (GT within +-48)")
    print("=" * 70)
    stats("(a) STAGE-1 idx (LLM generation)", err_s1)
    stats("(b) REFINE4 refined (windowed-trained, hard-argmax)", err_ref)
    stats("(c) FULL-SEQ MS-TCN read-out (gold, same events)", err_fs)


if __name__ == "__main__":
    main()
