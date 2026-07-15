"""Refine with an END-TO-END learned attention-pool over the 576-token grid, vs the
mean-pool baseline. Tests whether query-conditioned token pooling (trained jointly with
the MS-TCN) beats feat_interleave's spatial mean-pool. Same MS-TCN+FiLM backbone (Refine4),
same dense+softNMS eval, asymmetric input±W / emit±out_E. --pool {mean,attn,type}.
Only uses videos whose window grids were extracted (scripts/64)."""
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
LAB = os.path.join(_COMMON, "data", "HOI4D-v3")
CACHE = os.path.join(ROOT, "outputs/action/cache")
GRID = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/grid_win"
from refine4 import Refine4                                       # noqa: E402
from common.eval import non_maximum_supression                    # noqa: E402
import eval_nms as en                                             # noqa: E402


class GridPool(nn.Module):
    """attention-pool 576 tokens -> 768, query-conditioned. Conditioning source:
      mean : none (fixed mean-pool)
      attn : learned query only (no language)
      type : nn.Embedding(2) -- CLOSED VOCAB (touch/untouch index), not language-ready
      lang : LANGUAGE query vector (LLM hidden of the question, 4096) -> OPEN VOCAB.
             swap the question for any natural-language / referring / open-vocab query."""
    def __init__(self, d=768, heads=4, kind="attn", d_llm=4096):
        super().__init__()
        self.kind = kind; self.h = heads
        if kind != "mean":
            self.q = nn.Parameter(torch.randn(heads, d) * 0.02)
            self.kp = nn.Linear(d, d); self.vp = nn.Linear(d, d)
            self.o = nn.Linear(d, d)
            if kind == "type":
                self.temb = nn.Embedding(2, d)
            elif kind == "lang":
                self.lq = nn.Sequential(nn.Linear(d_llm, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, g, qlang, t):                               # g[B,L,576,768] qlang[B,4096] t[B]
        if self.kind == "mean":
            return g.mean(2)
        B, L, P, D = g.shape
        k = self.kp(g); v = self.vp(g)                            # [B,L,576,768]
        q = self.q.view(1, 1, self.h, D).expand(B, L, -1, -1)
        if self.kind == "type":
            q = q + self.temb(t).view(B, 1, 1, D)
        elif self.kind == "lang":
            q = q + self.lq(qlang).view(B, 1, 1, D)               # language-conditioned query
        a = torch.softmax((q @ k.transpose(-1, -2)) / (D ** 0.5), -1)   # [B,L,H,576]
        pooled = (a @ v).mean(2)                                   # [B,L,768]
        return self.o(pooled)


class GridRefine(nn.Module):
    def __init__(self, pool="attn", hid=128, n_layers=5, n_stages=3, n_lang=2):
        super().__init__()
        self.pool = GridPool(768, 4, pool)
        self.ref = Refine4(768, hid, n_layers, n_stages, n_lang, "film")

    def forward(self, g, q, m, t):
        x = self.pool(g, q, t)                                    # q = language query vector
        return self.ref(x, q, m)


def run(pool="attn", W=24, out_E=12, hid=128, dilate=1, fg=5.0, epochs=25, bs=32,
        lr=5e-4, run_name="rg", seed=0):
    torch.manual_seed(seed); dev = "cuda"; types = ("touch", "untouch"); L = 2 * W + 1
    tid = {"touch": 0, "untouch": 1}
    en.TOLS = [0, 1, 2]
    tq = torch.load(os.path.join(CACHE, "type_queries.pt"), weights_only=False)
    qv = {t: tq[t].to(dev) for t in types}
    have = set(os.path.splitext(f)[0] for f in os.listdir(GRID))
    gc = {}

    def gload(k):
        if k not in gc:
            d = np.load(os.path.join(GRID, k + ".npz"))
            fr = d["frames"].astype(np.int64); grids = d["grids"]           # (n,576,768) fp16
            pos = np.full(int(fr.max()) + 1, -1, np.int64); pos[fr] = np.arange(len(fr))
            gc[k] = (pos, grids)
        return gc[k]

    def gt_of(v, t):
        return sorted(e["frame"] for e in v["events"] if e["label"] == t)

    tr_coarse = torch.load(os.path.join(CACHE, "idx_multi_train_800.pt"), weights_only=False)["coarse"]
    va_coarse = torch.load(os.path.join(CACHE, "idx_multi_val_0.pt"), weights_only=False)["coarse"]
    trm = {v["video"].replace("/", "__"): v for v in json.load(open(os.path.join(LAB, "train.json")))}
    val_vids = [v for v in json.load(open(os.path.join(LAB, "val.json")))
                if v["video"].replace("/", "__") in have]

    pairs = []
    for (k, t), lst in tr_coarse.items():
        if k not in have or k not in trm:
            continue
        gts = gt_of(trm[k], t)
        for p, _ in lst:
            ing = [int(g - p) for g in gts if abs(g - p) <= out_E]
            pairs.append((k, t, int(p), tuple(ing) if ing else None))
    print(f"[cfg] refine_grid pool={pool} W={W} outE={out_E} | videos={len(have)} pairs={len(pairs)}", flush=True)

    def win(centers, ks, ts):
        B = len(centers)
        g = np.zeros((B, L, 576, 768), np.float16); m = np.zeros((B, L), np.float32)
        for i, (c, k) in enumerate(zip(centers, ks)):
            pos, grids = gload(k)
            fr = np.arange(c - W, c + W + 1)
            valid = (fr >= 0) & (fr < len(pos))
            rows = np.where(valid, pos[np.clip(fr, 0, len(pos) - 1)], -1)
            ok = rows >= 0
            g[i, ok] = grids[rows[ok]]; m[i, ok] = 1
        t = torch.tensor([tid[x] for x in ts])
        return torch.from_numpy(g).to(dev).float(), torch.from_numpy(m).unsqueeze(1).to(dev), t.to(dev)

    def labels(offs):
        y = torch.zeros(len(offs), L, dtype=torch.long)
        for i, o in enumerate(offs):
            if o is None:
                continue
            for oo in o:
                for j in range(max(0, W + oo - dilate), min(L, W + oo + dilate + 1)):
                    y[i, j] = 1
        return y.to(dev)

    model = GridRefine(pool, hid).to(dev)
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
        flat = [(v["video"], v["video"].replace("/", "__"), t, int(p))
                for v in val_vids for t in types for p, _ in va_coarse.get((v["video"].replace("/", "__"), t), [])]
        det = {v["video"]: [] for v in val_vids}
        for b0 in range(0, len(flat), 64):
            ch = flat[b0:b0 + 64]
            g, m, tt = win([c for *_, c in ch], [k for _, k, *_ in ch], [t for _, _, t, _ in ch])
            q = torch.stack([qv[t] for _, _, t, _ in ch])
            ev = F.softmax(model(g, q, m, tt)[-1], -1)[:, :, 1]
            evm = ev.masked_fill(m.squeeze(1) == 0, -1).cpu().numpy()
            for (vid, k, t, p), row in zip(ch, evm):
                for j in range(L):
                    if row[j] >= 0.01 and abs(j - W) <= out_E:
                        det[vid].append({"label": t, "frame": int(p - W + j), "score": float(row[j])})
        pl = [{"video": v["video"], "events": det[v["video"]]} for v in val_vids]
        return {"nonms": en.maps_quiet(truth, pl),
                "nms": en.maps_quiet(truth, non_maximum_supression(pl, 1)),
                "snms": en.maps_quiet(truth, en.soft_nms(pl, 4, 0.5))}

    out_dir = os.path.join(ROOT, "outputs", "spot", run_name); os.makedirs(out_dir, exist_ok=True)
    best = -1.0; bestr = None
    for ep in range(epochs):
        model.train(); rng = random.Random(1000 + ep); order = pairs[:]; rng.shuffle(order)
        rl, t0, nb = 0.0, time.time(), 0
        for b0 in range(0, spe * bs, bs):
            ch = order[b0:b0 + bs]
            g, m, tt = win([c for _, _, c, _ in ch], [k for k, _, _, _ in ch], [t for _, t, _, _ in ch])
            q = torch.stack([qv[t] for _, t, _, _ in ch]); y = labels([o for *_, o in ch])
            outs = model(g, q, m, tt)
            lm = m.squeeze(1).clone(); lm[:, :W - out_E] = 0; lm[:, W + out_E + 1:] = 0
            lmf = lm.reshape(-1); denom = lmf.sum().clamp(min=1)
            loss = sum((F.cross_entropy(outs[s].reshape(-1, 2), y.reshape(-1), weight=cew, reduction="none") * lmf).sum() / denom
                       for s in range(outs.shape[0]))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            rl += float(loss.detach()); nb += 1
        rr = evaluate(); s = rr["snms"]; crit = sum(s) / 3
        print(f"[ep{ep}] loss {rl/max(nb,1):.3f} | sNMS @0/1/2={[round(x,2) for x in s]} "
              f"(mean {crit:.2f}) noNMS@2={rr['nonms'][2]:.1f} | {len(order)/(time.time()-t0):.0f} it/s", flush=True)
        if crit > best:
            best = crit; bestr = rr
            torch.save({"model": model.state_dict(), "r": rr}, os.path.join(out_dir, "best.pt"))
    print(f"[done] {run_name} pool={pool} best(val,sNMS) @0/1/2={[round(x,2) for x in bestr['snms']]} "
          f"| nonms {[round(x,2) for x in bestr['nonms']]}", flush=True)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="lang", choices=["mean", "attn", "type", "lang"])
    ap.add_argument("--run_name", required=True); ap.add_argument("--W", type=int, default=24)
    ap.add_argument("--out_E", type=int, default=12); ap.add_argument("--epochs", type=int, default=25)
    a = ap.parse_args()
    run(pool=a.pool, W=a.W, out_E=a.out_E, epochs=a.epochs, run_name=a.run_name)
