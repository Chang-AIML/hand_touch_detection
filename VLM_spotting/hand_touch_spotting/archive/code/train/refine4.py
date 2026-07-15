"""Stage-2 refine v4 = refine2's PROVEN multi-stage MS-TCN backbone (recovers 55.6)
+ two requested improvements, cleanly ablatable:
  (③) language injection mode: 'film' (refine2, FiLM on input) vs 'xattn' (ADDED gated
      cross-attn residual layers on stage-1 hidden, gate init 0 -> base preserved).
  (④) reject: include pred-centric windows with NO GT inside -> all-0 label.
Both modes carry the TYPE query (touch/untouch), so the head is never type-blind.
Feature = RAW V-JEPA window. Eval: refine LLM val dets -> mAP@0/1/2.
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
from common.eval import non_maximum_supression                 # noqa: E402


class DRL(nn.Module):
    def __init__(self, d, c):
        super().__init__(); self.cd = nn.Conv1d(c, c, 3, padding=d, dilation=d); self.c1 = nn.Conv1d(c, c, 1); self.do = nn.Dropout()

    def forward(self, x, m):
        return (x + self.do(self.c1(F.relu(self.cd(x))))) * m


class LangResidual(nn.Module):
    """frames cross-attend the type query; gated residual (gate init 0 -> identity)."""
    def __init__(self, d, n_heads, d_llm):
        super().__init__()
        self.qp = nn.Sequential(nn.Linear(d_llm, d), nn.GELU(), nn.Linear(d, d))
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, h, q):                                    # h[B,L,d] q[B,d_llm]
        lang = self.qp(q).unsqueeze(1)
        a, _ = self.attn(self.norm(h), lang, lang)
        return h + self.gate * a


class Stage(nn.Module):
    def __init__(self, i, h, o, n_layers, n_lang=0, d_llm=4096, n_heads=4):
        super().__init__()
        self.c1 = nn.Conv1d(i, h, 1)
        self.L = nn.ModuleList([DRL(2 ** k, h) for k in range(n_layers)])
        self.lang = nn.ModuleList([LangResidual(h, n_heads, d_llm) for _ in range(n_lang)])
        self.co = nn.Conv1d(h, o, 1)

    def forward(self, x, q, m):
        h = self.c1(x.transpose(1, 2))
        for l in self.L:
            h = l(h, m)
        if self.lang:
            h = h.transpose(1, 2)
            for lg in self.lang:
                h = lg(h, q)
            h = h.transpose(1, 2)
        return (self.co(h) * m).transpose(1, 2)


class Refine4(nn.Module):
    def __init__(self, feat_dim=768, hid=128, n_layers=5, n_stages=3, n_lang=2,
                 lang_mode="xattn", d_llm=4096):
        super().__init__()
        self.lang_mode = lang_mode
        if lang_mode == "film":
            self.film = nn.Sequential(nn.Linear(d_llm, hid), nn.GELU(), nn.Linear(hid, 2 * feat_dim))
            nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)
            s1_lang = 0
        else:                                                  # xattn: gated cross-attn in stage 1
            s1_lang = n_lang
        self.s1 = Stage(feat_dim, hid, 2, n_layers, n_lang=s1_lang, d_llm=d_llm)
        self.ss = nn.ModuleList([Stage(2, hid, 2, n_layers, n_lang=0) for _ in range(n_stages - 1)])

    def forward(self, x, q, m):                                # x[B,L,768] q[B,d] m[B,1,L]
        if self.lang_mode == "film":
            g, b = self.film(q).chunk(2, -1)
            x = (1 + g).unsqueeze(1) * x + b.unsqueeze(1)
        out = self.s1(x, q, m); outs = [out]
        for s in self.ss:
            out = s(F.softmax(out, 2) * m.transpose(1, 2), q, m); outs.append(out)
        return torch.stack(outs)                               # [S,B,L,2]


def run(W=48, hid=128, n_layers=5, n_stages=3, n_lang=2, lang_mode="xattn", dilate=1,
        fg=5.0, epochs=30, bs=128, lr=5e-4, reject=True, multi_gt=False, centers="llm",
        out_E=None, pool_key="train_800", run_name="refine4", seed=0):
    """out_E: asymmetric input/output — input window ±W is context, but labels/loss/emission
    are restricted to the central ±out_E (outer frames NEVER supervised -> no contradiction)."""
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

    def recentre(coarse, seed_off=0):
        """centers ablation: replace LLM idx with random/grid positions (same counts)."""
        if centers == "llm":
            return coarse
        rng = random.Random(12345 + seed_off); out = {}
        for (k, t), lst in coarse.items():
            if not os.path.exists(os.path.join(F3, k + ".npy")):
                continue
            N = feats(k).shape[0]; n = len(lst)
            if n == 0:
                out[(k, t)] = []; continue
            if centers == "random":
                ps = sorted(rng.randrange(N) for _ in range(n))
            else:                                              # grid
                ps = [int((i + 0.5) * N / n) for i in range(n)]
            out[(k, t)] = [(int(p), float(sc)) for p, (_, sc) in zip(ps, lst)]
        return out

    tr_coarse = recentre(torch.load(os.path.join(CACHE, f"idx_multi_{pool_key}.pt"), weights_only=False)["coarse"], 0)
    va_coarse = recentre(torch.load(os.path.join(CACHE, "idx_multi_val_0.pt"), weights_only=False)["coarse"], 1)
    trm = {v["video"].replace("/", "__"): v for v in json.load(open(os.path.join(LAB, "train.json")))}
    val_vids = [v for v in json.load(open(os.path.join(LAB, "val.json")))
                if os.path.exists(os.path.join(F3, v["video"].replace("/", "__") + ".npy"))]

    pairs, n_pos, n_neg, n_multi = [], 0, 0, 0
    for (k, t), lst in tr_coarse.items():
        v = trm.get(k)
        if v is None:
            continue
        gts = gt_of(v, t)
        for p, _ in lst:
            ing = [int(g - p) for g in gts if abs(g - p) <= (out_E if out_E else W)]
            if ing:
                if multi_gt:                                  # label ALL same-type GTs in window
                    pairs.append((k, t, int(p), tuple(ing))); n_multi += len(ing) > 1
                else:                                         # legacy: nearest only
                    pairs.append((k, t, int(p), min(ing, key=abs)))
                n_pos += 1
            elif reject:
                pairs.append((k, t, int(p), None)); n_neg += 1
    print(f"[cfg] refine4 W={W} out_E={out_E} mode={lang_mode} n_lang={n_lang} stages={n_stages} "
          f"reject={reject} multi_gt={multi_gt} | pairs {len(pairs)} (pos {n_pos} / neg {n_neg} / multi {n_multi})", flush=True)

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
            if o is None:
                continue
            for oo in (o if isinstance(o, tuple) else (o,)):
                for j in range(max(0, W + oo - dilate), min(L, W + oo + dilate + 1)):
                    y[i, j] = 1
        return y.to(dev)

    model = Refine4(768, hid, n_layers, n_stages, n_lang, lang_mode).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR
    spe = len(pairs) // bs
    sched = ChainedScheduler([LinearLR(opt, 0.01, 1.0, 2 * spe), CosineAnnealingLR(opt, max(1, spe * (epochs - 2)))])
    cew = torch.tensor([1.0, fg], device=dev)
    truth = [{"video": v["video"], "num_frames": v["num_frames"],
              "events": [{"label": e["label"], "frame": e["frame"]} for e in v["events"]]} for v in val_vids]

    @torch.no_grad()
    def evaluate(coarse, vids, tru):
        """E2E-aligned: DENSE per-frame scored field per window -> {noNMS, NMS(1), softNMS(4,.5)}."""
        model.eval()
        flat = [(v["video"], v["video"].replace("/", "__"), t, int(p), float(sc))
                for v in vids for t in types for p, sc in coarse.get((v["video"].replace("/", "__"), t), [])]
        det = {v["video"]: [] for v in vids}
        for b0 in range(0, len(flat), 256):
            ch = flat[b0:b0 + 256]
            x, m = win([c for *_, c, _ in ch], [k for _, k, *_ in ch])
            q = torch.stack([qv[t] for _, _, t, _, _ in ch])
            ev = F.softmax(model(x, q, m)[-1], -1)[:, :, 1]
            evm = ev.masked_fill(m.squeeze(1) == 0, -1).cpu().numpy()
            for (vid, k, t, p, sc), row in zip(ch, evm):
                N = feats(k).shape[0]
                for j in range(L):
                    if row[j] >= 0.01:
                        if out_E is not None and abs(j - W) > out_E:
                            continue
                        f2 = p - W + j
                        if 0 <= f2 < N:
                            det[vid].append({"label": t, "frame": int(f2), "score": float(row[j])})
        pl = [{"video": v["video"], "events": det[v["video"]]} for v in vids]
        return {"nonms": en.maps_quiet(tru, pl),
                "nms": en.maps_quiet(tru, non_maximum_supression(pl, 1)),
                "snms": en.maps_quiet(tru, en.soft_nms(pl, 4, 0.5))}

    out_dir = os.path.join(ROOT, "outputs", "spot", run_name); os.makedirs(out_dir, exist_ok=True)
    mp = os.path.join(out_dir, "metrics.csv")
    open(mp, "w").write("epoch,loss,d0,d1,d2,n0,n1,n2,s0,s1,s2\n")
    best = -1.0; bestr = None
    for ep in range(epochs):
        model.train(); rng = random.Random(1000 + ep); order = pairs[:]; rng.shuffle(order)
        rl, t0, nb = 0.0, time.time(), 0
        for b0 in range(0, spe * bs, bs):
            ch = order[b0:b0 + bs]
            x, m = win([c for _, _, c, _ in ch], [k for k, _, _, _ in ch])
            q = torch.stack([qv[t] for _, t, _, _ in ch]); y = labels([o for *_, o in ch])
            outs = model(x, q, m)
            if out_E is None:
                loss = sum(F.cross_entropy((outs[s] * m.transpose(1, 2)).reshape(-1, 2), y.reshape(-1), weight=cew)
                           for s in range(outs.shape[0]))
            else:                                              # loss ONLY on the central ±out_E
                lm = m.squeeze(1).clone(); lm[:, :W - out_E] = 0; lm[:, W + out_E + 1:] = 0
                lmf = lm.reshape(-1); denom = lmf.sum().clamp(min=1)
                loss = sum((F.cross_entropy(outs[s].reshape(-1, 2), y.reshape(-1), weight=cew,
                                            reduction="none") * lmf).sum() / denom
                           for s in range(outs.shape[0]))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            rl += float(loss.detach()); nb += 1
        rr = evaluate(va_coarse, val_vids, truth)
        d, n, s = rr["nonms"], rr["nms"], rr["snms"]
        crit = sum(s) / 3                                       # select by soft-NMS mean(@0/1/2)
        print(f"[ep{ep}] loss {rl/max(nb,1):.3f} | sNMS @0/1/2={[round(x,2) for x in s]} "
              f"(mean {crit:.2f}) | noNMS@2={d[2]:.1f} nms@2={n[2]:.1f} | {len(order)/(time.time()-t0):.0f} it/s", flush=True)
        open(mp, "a").write(f"{ep},{rl/max(nb,1):.3f}," + ",".join(f"{x:.2f}" for x in (*d, *n, *s)) + "\n")
        if crit > best:
            best = crit; bestr = rr
            torch.save({"model": model.state_dict(), "r": rr}, os.path.join(out_dir, "best.pt"))
    print(f"[done] {run_name} best(val,sNMS) @0/1/2={[round(x,2) for x in bestr['snms']]}", flush=True)

    # ---- test eval on the best checkpoint: full E2E-style table ----
    test_path = os.path.join(CACHE, "idx_multi_test_0.pt")
    if os.path.exists(test_path):
        model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), weights_only=False)["model"])
        te_coarse = recentre(torch.load(test_path, weights_only=False)["coarse"], 2)
        te_vids = [v for v in json.load(open(os.path.join(LAB, "test.json")))
                   if os.path.exists(os.path.join(F3, v["video"].replace("/", "__") + ".npy"))]
        te_truth = [{"video": v["video"], "num_frames": v["num_frames"],
                     "events": [{"label": e["label"], "frame": e["frame"]} for e in v["events"]]} for v in te_vids]
        tr_ = evaluate(te_coarse, te_vids, te_truth)
        for name, key in (("without NMS", "nonms"), ("NMS (w=1)", "nms"), ("soft-NMS", "snms")):
            print(f"[TEST] {run_name} {name:<12} @0/1/2={[round(x,2) for x in tr_[key]]}", flush=True)
        open(mp, "a").write("TEST,0," + ",".join(f"{x:.2f}" for x in (*tr_['nonms'], *tr_['nms'], *tr_['snms'])) + "\n")
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True); ap.add_argument("--W", type=int, default=48)
    ap.add_argument("--n_stages", type=int, default=3); ap.add_argument("--n_lang", type=int, default=2)
    ap.add_argument("--hid", type=int, default=128); ap.add_argument("--n_layers", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lang_mode", default="xattn", choices=["film", "xattn"])
    ap.add_argument("--dilate", type=int, default=1); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_reject", action="store_true"); ap.add_argument("--multi_gt", action="store_true")
    ap.add_argument("--centers", default="llm", choices=["llm", "random", "grid"])
    ap.add_argument("--out_E", type=int, default=None)
    a = ap.parse_args()
    run(W=a.W, n_stages=a.n_stages, n_lang=a.n_lang, hid=a.hid, n_layers=a.n_layers,
        epochs=a.epochs, dilate=a.dilate, seed=a.seed, lang_mode=a.lang_mode,
        reject=not a.no_reject, multi_gt=a.multi_gt, centers=a.centers, out_E=a.out_E,
        run_name=a.run_name)
