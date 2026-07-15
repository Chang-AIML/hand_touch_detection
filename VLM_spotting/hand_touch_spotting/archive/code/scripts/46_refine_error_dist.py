"""2-STAGE localization-error figure: per-GT event error BEFORE (stage-1 LLM idx)
vs AFTER (stage-2 MS-TCN refine). Reuses cached val idx dets + the trained refine4
(r6_final, FiLM 3-stage W=48). Histogram + CDF overlay -> shows the refine pulling
the error mass toward 0. Saves error_dist_2stage.png (leaves the stage-1 png intact).
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
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "train"))
_COMMON = "/home/chang_noroot/data2/huyanh/Workspace/hand_touch_detection"
sys.path.insert(0, _COMMON)
LAB = os.path.join(_COMMON, "data", "HOI4D-v3")
F3 = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/feat_interleave"
CACHE = os.path.join(ROOT, "outputs/action/cache")
from refine4 import Refine4                                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "outputs/spot/r6_final/best.pt"))
    ap.add_argument("--split", default="val"); ap.add_argument("--W", type=int, default=48)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs/idx/idx_multi/error_dist_2stage.png"))
    args = ap.parse_args()
    dev = "cuda"; types = ("touch", "untouch"); W = args.W; L = 2 * W + 1

    tq = torch.load(os.path.join(CACHE, "type_queries.pt"), weights_only=False)
    qv = {t: tq[t].to(dev) for t in types}
    model = Refine4(768, 128, 5, 3, 2, "film").to(dev)
    model.load_state_dict(torch.load(args.ckpt, weights_only=False)["model"]); model.eval()
    cache = "idx_multi_val_0.pt" if args.split == "val" else f"idx_multi_{args.split}_0.pt"
    coarse = torch.load(os.path.join(CACHE, cache), weights_only=False)["coarse"]
    vids = [v for v in json.load(open(os.path.join(LAB, f"{args.split}.json")))
            if os.path.exists(os.path.join(F3, v["video"].replace("/", "__") + ".npy"))]
    fc = {}

    def feats(k):
        if k not in fc:
            fc[k] = np.load(os.path.join(F3, k + ".npy")).astype(np.float32)
        return fc[k]

    # stage-1 preds (from cache) and stage-2 refined preds (run refine model)
    s1 = {}; s2 = {}
    flat = [(v["video"].replace("/", "__"), t, int(p)) for v in vids for t in types
            for p, _ in coarse.get((v["video"].replace("/", "__"), t), [])]
    for k, t, p in flat:
        s1.setdefault((k, t), []).append(p)

    @torch.no_grad()
    def refine_batch(chunk):
        x = torch.zeros(len(chunk), L, 768); m = torch.zeros(len(chunk), L)
        for i, (k, t, p) in enumerate(chunk):
            f = feats(k); N = f.shape[0]
            for j in range(L):
                fr = p - W + j
                if 0 <= fr < N:
                    x[i, j] = torch.from_numpy(f[fr]); m[i, j] = 1
        x = x.to(dev); m = m.unsqueeze(1).to(dev)
        q = torch.stack([qv[t] for _, t, _ in chunk])
        ev = F.softmax(model(x, q, m)[-1], -1)[:, :, 1].masked_fill(m.squeeze(1) == 0, -1)
        off = (ev.argmax(-1) - W).cpu().numpy()
        return off

    for b0 in range(0, len(flat), 256):
        ch = flat[b0:b0 + 256]; offs = refine_batch(ch)
        for (k, t, p), o in zip(ch, offs):
            N = feats(k).shape[0]
            s2.setdefault((k, t), []).append(int(min(max(p + o, 0), N - 1)))

    # per-GT error for each stage
    def errs_of(preds):
        e = {"touch": [], "untouch": []}
        for v in vids:
            k = v["video"].replace("/", "__")
            for ev in v["events"]:
                t = ev["label"]; gt = ev["frame"]; ps = preds.get((k, t), [])
                e[t].append(min((abs(gt - p) for p in ps), default=999))
        return e

    e1, e2 = errs_of(s1), errs_of(s2)
    a1 = np.array(e1["touch"] + e1["untouch"]); a2 = np.array(e2["touch"] + e2["untouch"])

    def stats(a):
        h = a[a < 999]
        return {"n": len(a), "miss": 100 * (a >= 999).mean(), "median": np.median(h) if len(h) else 0,
                **{f"w{k}": 100 * (a <= k).mean() for k in (0, 1, 2, 4, 8, 16)}}

    S1, S2 = stats(a1), stats(a2)
    print("stage-1 (LLM idx):", {k: round(v, 1) for k, v in S1.items()})
    print("stage-2 (refined):", {k: round(v, 1) for k, v in S2.items()})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (axh, axc) = plt.subplots(1, 2, figsize=(13, 4.6))
    bins = np.arange(0, 32)
    axh.hist(np.clip(a1[a1 < 999], 0, 31), bins=bins, alpha=0.5, color="#9aa0a6",
             label=f"stage-1 LLM idx  (median {S1['median']:.0f}f)")
    axh.hist(np.clip(a2[a2 < 999], 0, 31), bins=bins, alpha=0.75, color="#d1495b",
             label=f"stage-2 + refine  (median {S2['median']:.0f}f)")
    axh.set_xlabel("localization error |pred - GT| (frames)"); axh.set_ylabel("count")
    axh.set_title("Per-event error: mass pulled toward 0 by the refine"); axh.legend()

    xs = np.arange(0, 31)
    axc.plot(xs, [100 * (a1 <= x).mean() for x in xs], color="#9aa0a6", lw=2.2, label="stage-1 LLM idx")
    axc.plot(xs, [100 * (a2 <= x).mean() for x in xs], color="#d1495b", lw=2.2, label="stage-2 + refine")
    for x in (0, 1, 2, 4, 8, 16):
        axc.axvline(x, color="gray", ls=":", lw=0.6)
    axc.set_xlabel("tolerance (frames)"); axc.set_ylabel("% GT events within tolerance")
    axc.set_title("Cumulative recall vs tolerance"); axc.grid(alpha=0.3); axc.legend(loc="lower right")
    txt = (f"within  1f: {S1['w1']:.0f}% -> {S2['w1']:.0f}%\n"
           f"within  2f: {S1['w2']:.0f}% -> {S2['w2']:.0f}%\n"
           f"within  4f: {S1['w4']:.0f}% -> {S2['w4']:.0f}%")
    axc.text(0.03, 0.97, txt, transform=axc.transAxes, va="top", fontsize=9,
             family="monospace", bbox=dict(boxstyle="round", fc="#fff6f7", ec="#d1495b"))
    fig.suptitle(f"LLM idx -> MS-TCN refine (2-stage) — localization error ({args.split}, "
                 f"stage-1 mAP@2 44.9 -> 2-stage 56.7)", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"[plot] {args.out}")


if __name__ == "__main__":
    main()
