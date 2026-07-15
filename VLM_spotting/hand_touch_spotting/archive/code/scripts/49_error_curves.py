"""Compare per-event localization ERROR curves (same metric as outputs/idx/idx_multi/
error_dist.png: each GT event -> |gt - nearest same-type prediction|, hist + CDF) for:
  stage-1 LLM idx (cache) vs refine W=8+rej vs W=12+rej vs W=48.
Saves plot/refine_error_dist_{val,test}.png."""
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

dev = "cuda"; types = ("touch", "untouch")
tq = torch.load(os.path.join(CACHE, "type_queries.pt"), weights_only=False)
qv = {t: tq[t].to(dev) for t in types}
fc = {}

CONFIGS = [                                                      # label, run_dir(None=stage-1), W, color
    ("stage-1 LLM idx", None,         0,  "#999999"),
    ("refine ±8 +rej",  "r7_w8_rej",  8,  "#e8a63d"),
    ("refine ±12 +rej", "r7_w12_rej", 12, "#2e86ab"),
    ("refine ±48",      "r6_final",   48, "#d1495b"),
]


def feats(k):
    if k not in fc:
        fc[k] = np.load(os.path.join(F3, k + ".npy")).astype(np.float32)
    return fc[k]


@torch.no_grad()
def refined_preds(model, W, coarse, vids):
    """per (vid_key,type) -> list of refined frames (peak argmax per idx)."""
    L = 2 * W + 1; out = {}
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
            off = evm.argmax(-1) - W
            out[(k, t)] = [int(min(max(c + o, 0), N - 1)) for c, o in zip(centers, off)]
    return out


def per_event_errors(preds, vids):
    errs = []
    for v in vids:
        k = v["video"].replace("/", "__")
        for e in v["events"]:
            ps = preds.get((k, e["label"]), [])
            errs.append(min((abs(e["frame"] - p) for p in ps), default=999))
    return np.array(errs)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(ROOT, "plot"), exist_ok=True)

    for split in ("val", "test"):
        coarse = torch.load(os.path.join(CACHE, f"idx_multi_{split}_0.pt"), weights_only=False)["coarse"]
        vids = [v for v in json.load(open(os.path.join(LAB, f"{split}.json")))
                if os.path.exists(os.path.join(F3, v["video"].replace("/", "__") + ".npy"))]
        curves = []
        print(f"\n===== {split} — per-GT-event error (to nearest same-type pred) =====")
        print(f"{'config':<18} {'miss%':>6} {'med':>5} | {'=0':>6} {'<=1':>6} {'<=2':>6} {'<=4':>6} {'<=8':>6}")
        for label, run, W, color in CONFIGS:
            if run is None:
                preds = {kt: [int(p) for p, _ in lst] for kt, lst in coarse.items()}
            else:
                model = Refine4(768, 128, 5, 3, 2, "film").to(dev)
                model.load_state_dict(torch.load(os.path.join(ROOT, "outputs/spot", run, "best.pt"),
                                                 weights_only=False)["model"]); model.eval()
                preds = refined_preds(model, W, coarse, vids)
            e = per_event_errors(preds, vids)
            hit = e < 999
            print(f"{label:<18} {100*(~hit).mean():>5.1f}% {np.median(e[hit]):>5.1f} | "
                  + " ".join(f"{100*(e <= k).mean():>5.1f}%" for k in (0, 1, 2, 4, 8)), flush=True)
            curves.append((label, color, e))

        fig, (axh, axc) = plt.subplots(1, 2, figsize=(13.5, 4.6))
        bins = np.arange(0, 32)
        for label, color, e in curves:
            a = e[e < 999]
            axh.hist(a, bins=bins, histtype="step", lw=2, color=color, label=label, density=True)
            xs = np.arange(0, 31)
            axc.plot(xs, [100 * (e <= x).mean() for x in xs], lw=2.2, color=color,
                     label=f"{label}  (=0: {100*(e==0).mean():.0f}%, ≤2: {100*(e<=2).mean():.0f}%)")
        axh.set_xlabel("|error| frames"); axh.set_ylabel("density"); axh.set_title(f"{split}: per-event error histogram")
        axc.set_xlabel("tolerance (frames)"); axc.set_ylabel("% GT events covered"); axc.set_ylim(0, 100)
        axc.set_title(f"{split}: coverage CDF (higher=better)")
        axc.axvline(2, color="k", ls=":", lw=0.8)
        for ax in (axh, axc):
            ax.legend(fontsize=8.5); ax.grid(alpha=0.25)
        out = os.path.join(ROOT, "plot", f"refine_error_dist_{split}.png")
        fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
