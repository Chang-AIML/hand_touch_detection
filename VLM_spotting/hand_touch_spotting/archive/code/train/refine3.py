"""Stage-2 refine v3 — incorporates the requested improvements:
  (3) base MS-TCN localization path UNCHANGED; language injected via ADDED gated
      cross-attn residual layers (gate init 0 -> starts as identity -> won't hurt base).
  (4) train on pred-centric windows INCLUDING those with NO GT inside -> label all-0
      (rejection: idx with no real event nearby -> low score -> suppressed).
  (1) window W configurable; (binary head bg/event, query-conditioned).
Feature = RAW V-JEPA window (best per ablation). Eval: refine LLM val dets -> mAP@0/1/2.
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
sys.path.insert(0, _COMMON); sys.path.insert(0, os.path.join(_COMMON, "methods/spot_head"))
F3 = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/feat_interleave"
CACHE = os.path.join(ROOT, "outputs/action/cache")
LAB = os.path.join(_COMMON, "data", "HOI4D-v3")
import eval_nms as en                                          # noqa: E402


class DRL(nn.Module):
    def __init__(self, d, c):
        super().__init__(); self.cd = nn.Conv1d(c, c, 3, padding=d, dilation=d); self.c1 = nn.Conv1d(c, c, 1); self.do = nn.Dropout()

    def forward(self, x, m):
        return (x + self.do(self.c1(F.relu(self.cd(x))))) * m


class LangResidual(nn.Module):
    """frames cross-attend the query; gated residual (gate init 0 -> identity at start)."""
    def __init__(self, d, n_heads, d_llm):
        super().__init__()
        self.qp = nn.Sequential(nn.Linear(d_llm, d), nn.GELU(), nn.Linear(d, d))
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, h, q, pad):                               # h[B,L,d] q[B,d_llm] pad[B,L] True=pad
        lang = self.qp(q).unsqueeze(1)                          # [B,1,d]
        a, _ = self.attn(self.norm(h), lang, lang, key_padding_mask=None)
        return h + self.gate * a


class Refine3(nn.Module):
    def __init__(self, feat_dim=768, hid=128, n_layers=6, n_lang=2, n_heads=4, d_llm=4096):
        super().__init__()
        self.inp = nn.Conv1d(feat_dim, hid, 1)
        self.tcn = nn.ModuleList([DRL(2 ** i, hid) for i in range(n_layers)])   # base localization (unchanged)
        self.lang = nn.ModuleList([LangResidual(hid, n_heads, d_llm) for _ in range(n_lang)])  # ADDED
        self.head = nn.Conv1d(hid, 2, 1)

    def forward(self, x, q, m):                                 # x[B,L,768] q[B,d_llm] m[B,1,L]
        h = self.inp(x.transpose(1, 2))
        for l in self.tcn:
            h = l(h, m)
        h = h.transpose(1, 2)                                   # [B,L,hid]
        pad = (m.squeeze(1) == 0)
        for lg in self.lang:
            h = lg(h, q, pad)
        return (self.head(h.transpose(1, 2)) * m).transpose(1, 2)   # [B,L,2]


def run(W=48, hid=128, n_layers=6, n_lang=2, dilate=1, fg=5.0, epochs=30, bs=128, lr=5e-4,
        reject=True, pool_key="train_800", run_name="refine3", seed=0):
    torch.manual_seed(seed); dev = "cuda"; types = ("touch", "untouch"); L = 2 * W + 1
    en.TOLS = [0, 1, 2]
    tq = torch.load(os.path.join(CACHE, "type_queries.pt"), weights_only=False)
    qv = {t: tq[t].to(dev) for t in types}
    fc = {}

    def feats(k):
        if k not in fc:
            fc[k] = np.load(os.path.join(F3, k + ".npy")).astype(np.float32)
        return fc[k]

    def gt_of(v, t):
        return sorted(e["frame"] for e in v["events"] if e["label"] == t)

    tr_coarse = torch.load(os.path.join(CACHE, f"idx_multi_{pool_key}.pt"), weights_only=False)["coarse"]
    va_coarse = torch.load(os.path.join(CACHE, "idx_multi_val_0.pt"), weights_only=False)["coarse"]
    trm = {v["video"].replace("/", "__"): v for v in json.load(open(os.path.join(LAB, "train.json")))}
    val_vids = [v for v in json.load(open(os.path.join(LAB, "val.json")))
                if os.path.exists(os.path.join(F3, v["video"].replace("/", "__") + ".npy"))]

    # PRED-CENTRIC pairs; off=None means NO GT in window -> all-0 (reject)
    pairs, n_pos, n_neg = [], 0, 0
    for (k, t), lst in tr_coarse.items():
        v = trm.get(k)
        if v is None:
            continue
        gts = gt_of(v, t)
        for p, _ in lst:
            off = None
            if gts:
                o = min(gts, key=lambda x: abs(x - p)) - p
                if abs(o) <= W:
                    off = int(o)
            if off is not None:
                pairs.append((k, t, int(p), off)); n_pos += 1
            elif reject:
                pairs.append((k, t, int(p), None)); n_neg += 1
    print(f"[cfg] refine3 W={W} n_lang={n_lang} reject={reject} | pairs {len(pairs)} "
          f"(pos {n_pos} / neg {n_neg})", flush=True)

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
            if o is not None:
                for j in range(max(0, W + o - dilate), min(L, W + o + dilate + 1)):
                    y[i, j] = 1
        return y.to(dev)

    model = Refine3(768, hid, n_layers, n_lang).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR
    spe = len(pairs) // bs
    sched = ChainedScheduler([LinearLR(opt, 0.01, 1.0, 2 * spe), CosineAnnealingLR(opt, max(1, spe * (epochs - 2)))])
    cew = torch.tensor([1.0, fg], device=dev)
    truth = [{"video": v["video"], "num_frames": v["num_frames"],
              "events": [{"label": e["label"], "frame": e["frame"]} for e in v["events"]]} for v in val_vids]

    @torch.no_grad()
    def evaluate():
        model.eval()
        flat = [(v["video"], v["video"].replace("/", "__"), t, int(p), float(sc))
                for v in val_vids for t in types for p, sc in va_coarse.get((v["video"].replace("/", "__"), t), [])]
        det = {v["video"]: [] for v in val_vids}; base = {v["video"]: [] for v in val_vids}
        for b0 in range(0, len(flat), 256):
            ch = flat[b0:b0 + 256]
            x, m = win([c for *_, c, _ in ch], [k for _, k, *_ in ch])
            q = torch.stack([qv[t] for _, _, t, _, _ in ch])
            ev = F.softmax(model(x, q, m), -1)[:, :, 1]
            ev = ev.masked_fill(m.squeeze(1) == 0, -1)
            off = (ev.argmax(-1) - W).cpu().numpy(); pk = ev.max(-1).values.cpu().numpy()
            for (vid, k, t, p, sc), o, pp in zip(ch, off, pk):
                N = feats(k).shape[0]
                det[vid].append({"label": t, "frame": int(min(max(p + o, 0), N - 1)), "score": float(pp)})
                base[vid].append({"label": t, "frame": int(p), "score": float(sc)})
        pl = [{"video": v["video"], "events": det[v["video"]]} for v in val_vids]
        bl = [{"video": v["video"], "events": base[v["video"]]} for v in val_vids]
        return en.maps_quiet(truth, pl), en.maps_quiet(truth, bl)

    out_dir = os.path.join(ROOT, "outputs", "spot", run_name); os.makedirs(out_dir, exist_ok=True)
    mp = os.path.join(out_dir, "metrics.csv"); open(mp, "w").write("epoch,loss,r0,r1,r2,b0,b1,b2\n")
    best = -1.0; bb = [0, 0, 0]
    for ep in range(epochs):
        model.train(); rng = random.Random(1000 + ep); order = pairs[:]; rng.shuffle(order)
        rl, t0, nb = 0.0, time.time(), 0
        for b0 in range(0, spe * bs, bs):
            ch = order[b0:b0 + bs]
            x, m = win([c for _, _, c, _ in ch], [k for k, _, _, _ in ch])
            q = torch.stack([qv[t] for _, t, _, _ in ch]); y = labels([o for *_, o in ch])
            out = model(x, q, m)
            loss = F.cross_entropy((out * m.transpose(1, 2)).reshape(-1, 2), y.reshape(-1), weight=cew)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            rl += float(loss.detach()); nb += 1
        rr, bb = evaluate()
        print(f"[ep{ep}] loss {rl/max(nb,1):.3f} | REFINED @0/1/2={[round(x,2) for x in rr]} "
              f"| base {[round(x,2) for x in bb]} | {len(order)/(time.time()-t0):.0f} it/s", flush=True)
        open(mp, "a").write(f"{ep},{rl/max(nb,1):.3f},{rr[0]:.2f},{rr[1]:.2f},{rr[2]:.2f},{bb[0]:.2f},{bb[1]:.2f},{bb[2]:.2f}\n")
        if rr[2] > best:
            best = rr[2]; torch.save({"model": model.state_dict(), "r": rr}, os.path.join(out_dir, "best.pt"))
    print(f"[done] {run_name} best REFINED @2 {best:.2f} (stage-1 base @2 {bb[2]:.2f})", flush=True)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True); ap.add_argument("--W", type=int, default=48)
    ap.add_argument("--n_lang", type=int, default=2); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--no_reject", action="store_true")
    a = ap.parse_args()
    run(W=a.W, n_lang=a.n_lang, epochs=a.epochs, reject=not a.no_reject, run_name=a.run_name)
