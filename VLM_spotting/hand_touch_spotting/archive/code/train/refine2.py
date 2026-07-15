"""Stage-2 refine done right (keeps the LLM-2-stage design): for each stage-1 LLM
<idx>, run a LOCAL MS-TCN on the RAW V-JEPA window (±W) around it + FiLM(query),
E2E hard-label per-frame CE (bg/event, no Gaussian), argmax -> refined frame.

Fixes vs the earlier refiner: (1) window feature = RAW V-JEPA (best per ablation), NOT
the pre-LLM adaptor output (worst); (2) hard-label CE + argmax, not Gaussian soft-argmax.
Train = PRED-CENTRIC pairs from the frozen LLM idx generation (center=idx, target=gt-idx,
|.|<=W). Eval = refine the LLM's val detections -> mAP@0/1/2 (+ NMS variants).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
_COMMON = "/home/chang_noroot/data2/huyanh/Workspace/hand_touch_detection"
sys.path.insert(0, _COMMON)
sys.path.insert(0, os.path.join(_COMMON, "methods/spot_head"))
F3 = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/feat_interleave"
CACHE = os.path.join(ROOT, "outputs/action/cache")
from common.score import compute_mAPs                            # noqa: E402
from common.eval import non_maximum_supression                  # noqa: E402
import eval_nms as en                                           # noqa: E402


class TCN(nn.Module):
    class DRL(nn.Module):
        def __init__(self, d, c):
            super().__init__(); self.cd = nn.Conv1d(c, c, 3, padding=d, dilation=d); self.c1 = nn.Conv1d(c, c, 1); self.do = nn.Dropout()

        def forward(self, x, m):
            return (x + self.do(self.c1(F.relu(self.cd(x))))) * m

    def __init__(self, i, h, o, n):
        super().__init__(); self.c1 = nn.Conv1d(i, h, 1)
        self.L = nn.ModuleList([TCN.DRL(2 ** k, h) for k in range(n)]); self.co = nn.Conv1d(h, o, 1)

    def forward(self, x, m):
        x = self.c1(x.transpose(1, 2))
        for l in self.L:
            x = l(x, m)
        return (self.co(x) * m).transpose(1, 2)


class LocalMSTCN(nn.Module):
    def __init__(self, feat_dim=768, hid=128, n_layers=5, n_stages=2, d_llm=4096):
        super().__init__()
        self.film = nn.Sequential(nn.Linear(d_llm, hid), nn.GELU(), nn.Linear(hid, 2 * feat_dim))
        nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)
        self.s1 = TCN(feat_dim, hid, 2, n_layers)
        self.ss = nn.ModuleList([TCN(2, hid, 2, n_layers) for _ in range(n_stages - 1)])

    def forward(self, x, q, m):                                  # x[B,L,768] q[B,d] m[B,1,L]
        g, b = self.film(q).chunk(2, -1)
        x = (1 + g).unsqueeze(1) * x + b.unsqueeze(1)
        out = self.s1(x, m); outs = [out]
        for s in self.ss:
            out = s(F.softmax(out, 2) * m.transpose(1, 2), m); outs.append(out)
        return torch.stack(outs)                                 # [S,B,L,2]


def run(W=24, hid=128, n_layers=5, n_stages=2, dilate=1, fg=5.0, epochs=30, bs=128,
        lr=5e-4, pool_key="train_800", run_name="refine2", nms_win=2, seed=0):
    torch.manual_seed(seed); dev = "cuda"; types = ("touch", "untouch")
    L = 2 * W + 1
    tq = torch.load(os.path.join(CACHE, "type_queries.pt"), weights_only=False)
    qv = {t: tq[t].to(dev) for t in types}
    LAB = os.path.join(_COMMON, "data", "HOI4D-v3")
    fc = {}

    def feats(k):
        if k not in fc:
            fc[k] = np.load(os.path.join(F3, k + ".npy")).astype(np.float32)
        return fc[k]

    def gt_of(v, t):
        return sorted(e["frame"] for e in v["events"] if e["label"] == t)

    # stage-1 idx caches (keys are video__nobs)
    tr_coarse = torch.load(os.path.join(CACHE, f"idx_multi_{pool_key}.pt"), weights_only=False)["coarse"]
    va_coarse = torch.load(os.path.join(CACHE, "idx_multi_val_0.pt"), weights_only=False)["coarse"]
    val_vids = json.load(open(os.path.join(LAB, "val.json")))
    val_vids = [v for v in val_vids if os.path.exists(os.path.join(F3, v["video"].replace("/", "__") + ".npy"))]
    tr_vids = json.load(open(os.path.join(LAB, "train.json")))
    trm = {v["video"].replace("/", "__"): v for v in tr_vids}

    # PRED-CENTRIC pairs
    pairs = []
    for (k, t), lst in tr_coarse.items():
        v = trm.get(k)
        if v is None:
            continue
        gts = gt_of(v, t)
        for p, _ in lst:
            if gts:
                off = min(gts, key=lambda x: abs(x - p)) - p
                if abs(off) <= W:
                    pairs.append((k, t, int(p), int(off)))
    print(f"[cfg] refine2 W={W} raw-VJEPA hid={hid} L={n_layers} stages={n_stages} "
          f"dilate={dilate} | pairs {len(pairs)}", flush=True)

    def win(centers, ks):
        x = torch.zeros(len(centers), L, 768); m = torch.zeros(len(centers), L)
        for i, (c, k) in enumerate(zip(centers, ks)):
            f = feats(k); N = f.shape[0]
            for j in range(L):
                fr = c - W + j
                if 0 <= fr < N:
                    x[i, j] = torch.from_numpy(f[fr]); m[i, j] = 1
        return x.to(dev), m.unsqueeze(1).to(dev)

    def labels(offs):
        y = torch.zeros(len(offs), L, dtype=torch.long)
        for i, o in enumerate(offs):
            for j in range(max(0, W + o - dilate), min(L, W + o + dilate + 1)):
                y[i, j] = 1
        return y.to(dev)

    model = LocalMSTCN(768, hid, n_layers, n_stages).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR
    spe = len(pairs) // bs
    sched = ChainedScheduler([LinearLR(opt, 0.01, 1.0, 2 * spe), CosineAnnealingLR(opt, spe * (epochs - 2))])
    cew = torch.tensor([1.0, fg], device=dev)

    truth = [{"video": v["video"], "num_frames": v["num_frames"],
              "events": [{"label": e["label"], "frame": e["frame"]} for e in v["events"]]} for v in val_vids]
    en.TOLS = [0, 1, 2]

    @torch.no_grad()
    def evaluate():
        model.eval()
        flat = []
        for v in val_vids:
            k = v["video"].replace("/", "__")
            for t in types:
                for p, sc in va_coarse.get((k, t), []):
                    flat.append((v["video"], k, t, int(p), float(sc)))
        det = {v["video"]: [] for v in val_vids}
        for b0 in range(0, len(flat), 256):
            ch = flat[b0:b0 + 256]
            x, m = win([c for *_, c, _ in ch], [k for _, k, *_ in ch])
            q = torch.stack([qv[t] for _, _, t, _, _ in ch])
            ev = F.softmax(model(x, q, m)[-1], -1)[:, :, 1]                    # [B,L] event prob
            ev = ev.masked_fill(m.squeeze(1) == 0, -1)
            off = (ev.argmax(-1) - W).cpu().numpy()
            pk = ev.max(-1).values.cpu().numpy()
            for (vid, k, t, p, sc), o, pp in zip(ch, off, pk):
                N = feats(k).shape[0]
                det[vid].append({"label": t, "frame": int(min(max(p + o, 0), N - 1)), "score": float(pp)})
        base = {v["video"]: [] for v in val_vids}
        for v in val_vids:
            k = v["video"].replace("/", "__")
            for t in types:
                for p, sc in va_coarse.get((k, t), []):
                    base[v["video"]].append({"label": t, "frame": int(p), "score": float(sc)})
        pl = [{"video": v["video"], "events": det[v["video"]]} for v in val_vids]
        bl = [{"video": v["video"], "events": base[v["video"]]} for v in val_vids]
        rr = en.maps_quiet(truth, pl); bb = en.maps_quiet(truth, bl)
        return rr, bb

    out_dir = os.path.join(ROOT, "outputs", "spot", run_name); os.makedirs(out_dir, exist_ok=True)
    mp = os.path.join(out_dir, "metrics.csv"); open(mp, "w").write("epoch,loss,r0,r1,r2,b0,b1,b2\n")
    best = -1.0
    for ep in range(epochs):
        model.train(); rng = random.Random(1000 + ep); order = pairs[:]; rng.shuffle(order)
        rl, t0, nb = 0.0, time.time(), 0
        for b0 in range(0, spe * bs, bs):
            ch = order[b0:b0 + bs]
            x, m = win([c for _, _, c, _ in ch], [k for k, _, _, _ in ch])
            q = torch.stack([qv[t] for _, t, _, _ in ch])
            y = labels([o for *_, o in ch])
            outs = model(x, q, m)
            loss = sum(F.cross_entropy((outs[s] * m.transpose(1, 2)).reshape(-1, 2), y.reshape(-1), weight=cew)
                       for s in range(outs.shape[0]))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            rl += float(loss.detach()); nb += 1
        rr, bb = evaluate()
        print(f"[ep{ep}] loss {rl/max(nb,1):.3f} | REFINED @0/1/2 = {[round(x,2) for x in rr]} "
              f"| base(stage1) {[round(x,2) for x in bb]} | {len(order)/(time.time()-t0):.0f} it/s", flush=True)
        open(mp, "a").write(f"{ep},{rl/max(nb,1):.3f},{rr[0]:.2f},{rr[1]:.2f},{rr[2]:.2f},{bb[0]:.2f},{bb[1]:.2f},{bb[2]:.2f}\n")
        if rr[2] > best:
            best = rr[2]; torch.save({"model": model.state_dict(), "r": rr, "b": bb}, os.path.join(out_dir, "best.pt"))
    print(f"[done] {run_name} best REFINED @2 {best:.2f} (stage-1 base @2 {bb[2]:.2f})", flush=True)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="refine2"); ap.add_argument("--W", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--dilate", type=int, default=1)
    ap.add_argument("--n_stages", type=int, default=2)
    a = ap.parse_args()
    run(W=a.W, epochs=a.epochs, dilate=a.dilate, n_stages=a.n_stages, run_name=a.run_name)
