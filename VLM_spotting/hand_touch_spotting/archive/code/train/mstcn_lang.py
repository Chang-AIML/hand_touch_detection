"""Language-conditioned MS-TCN (用法 A): their SingleStageTCN (E2E-Spot head) with a
minimal FiLM injection from an LLM type-query. Query-conditioned binary spotting:
query (touch/untouch) -> FiLM(input feats) -> per-frame {bg, event}, fg-weighted CE.

--lang off  => plain query-conditioned MS-TCN (no FiLM) — the "harmless" baseline in
the SAME setup. --lang on => + FiLM. If lang ≈ no-lang, the language injection is
harmless; the machinery is then ready for future referring / open-vocab queries.
Full-video (300f) train+eval; mAP via the shared scorer (comparable to 67.8).
"""
from __future__ import annotations

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
FEAT = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/feat_interleave"
LAB = os.path.join(_COMMON, "data", "HOI4D-v3")
from common.score import compute_mAPs                          # noqa: E402


class SingleStageTCN(nn.Module):                               # copied from their common.py
    class DRL(nn.Module):
        def __init__(self, dilation, c):
            super().__init__()
            self.conv_dilated = nn.Conv1d(c, c, 3, padding=dilation, dilation=dilation)
            self.conv_1x1 = nn.Conv1d(c, c, 1); self.dropout = nn.Dropout()

        def forward(self, x, m):
            out = self.conv_1x1(F.relu(self.conv_dilated(x)))
            return (x + self.dropout(out)) * m[:, 0:1, :]

    def __init__(self, in_dim, hid, out_dim, n_layers, dilate=True):
        super().__init__()
        self.conv_1x1 = nn.Conv1d(in_dim, hid, 1)
        self.layers = nn.ModuleList([SingleStageTCN.DRL(2 ** i if dilate else 1, hid)
                                     for i in range(n_layers)])
        self.conv_out = nn.Conv1d(hid, out_dim, 1)

    def forward(self, x, m):                                   # x [B,L,C]
        m = m.permute(0, 2, 1)
        x = self.conv_1x1(x.permute(0, 2, 1))
        for lyr in self.layers:
            x = lyr(x, m)
        return (self.conv_out(x) * m[:, 0:1, :]).permute(0, 2, 1)


class FiLMMSTCN(nn.Module):
    def __init__(self, feat_dim=768, num_classes=2, num_stages=3, hid=256, n_layers=5,
                 d_llm=4096, use_lang=True):
        super().__init__()
        self.use_lang = use_lang
        self.stage1 = SingleStageTCN(feat_dim, hid, num_classes, n_layers)
        self.stages = nn.ModuleList([SingleStageTCN(num_classes, hid, num_classes, n_layers)
                                     for _ in range(num_stages - 1)])
        if use_lang:
            self.film = nn.Sequential(nn.Linear(d_llm, hid), nn.GELU(), nn.Linear(hid, 2 * feat_dim))
            nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)   # start = identity

    def forward(self, x, q, m):                                # x[B,L,768] q[B,d_llm] m[B,L,1]
        if self.use_lang:
            g, b = self.film(q).chunk(2, dim=-1)               # [B,768]
            x = (1 + g).unsqueeze(1) * x + b.unsqueeze(1)      # FiLM (identity at init)
        out = self.stage1(x, m); outs = [out]
        for s in self.stages:
            out = s(F.softmax(out, dim=2) * m, m); outs.append(out)
        return torch.stack(outs)                               # [S,B,L,2]


def load_split(split):
    return json.load(open(os.path.join(LAB, f"{split}.json")))


def build_items(vids, types=("touch", "untouch")):
    items = []
    for v in vids:
        if not os.path.exists(os.path.join(FEAT, v["video"].replace("/", "__") + ".npy")):
            continue
        for t in types:
            if any(e["label"] == t for e in v["events"]):
                items.append((v, t))
    return items


def run(lang=True, num_stages=3, hid=256, n_layers=5, dilate_lbl=4, fg_weight=5.0,
        epochs=50, batch_size=16, lr=1e-3, run_name="mstcn_lang", seed=0):
    torch.manual_seed(seed); dev = "cuda"; types = ("touch", "untouch")
    tq = torch.load(os.path.join(ROOT, "outputs/action/cache/type_queries.pt"), weights_only=False)
    qvec = {t: tq[t].to(dev) for t in types}

    fc = {}

    def feats(v):
        k = v.replace("/", "__")
        if k not in fc:
            fc[k] = np.load(os.path.join(FEAT, k + ".npy")).astype(np.float32)
        return fc[k]

    tr_items = build_items(load_split("train"))
    val_vids = load_split("val")
    print(f"[cfg] FiLM-MSTCN lang={lang} stages={num_stages} | train items {len(tr_items)} "
          f"| val vids {len(val_vids)}", flush=True)

    model = FiLMMSTCN(768, 2, num_stages, hid, n_layers, use_lang=lang).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = (len(tr_items) // batch_size) * epochs
    from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR
    sched = ChainedScheduler([LinearLR(opt, 0.01, 1.0, 3 * (len(tr_items) // batch_size)),
                              CosineAnnealingLR(opt, steps)])
    ce_w = torch.tensor([1.0, fg_weight], device=dev)

    def make_label(v, t):                                      # per-frame binary for type t
        N = v["num_frames"]; y = np.zeros(N, dtype=np.int64)
        for e in v["events"]:
            if e["label"] == t:
                for i in range(max(0, e["frame"] - dilate_lbl), min(N, e["frame"] + dilate_lbl + 1)):
                    y[i] = 1
        return y

    @torch.no_grad()
    def evaluate():
        model.eval()
        truth = [{"video": v["video"], "num_frames": v["num_frames"],
                  "events": [{"label": e["label"], "frame": e["frame"]} for e in v["events"]]}
                 for v in val_vids if os.path.exists(os.path.join(FEAT, v["video"].replace("/", "__") + ".npy"))]
        pred = []
        for v in truth:
            f = feats(v["video"]); N = f.shape[0]
            x = torch.from_numpy(f).unsqueeze(0).to(dev); m = torch.ones(1, N, 1, device=dev)
            evs = []
            for t in types:
                out = model(x, qvec[t].unsqueeze(0), m)[-1]     # last stage [1,N,2]
                p = F.softmax(out[0], dim=-1)[:, 1].cpu().numpy()  # event prob
                for i in range(N):
                    if p[i] >= 0.01:
                        evs.append({"label": t, "frame": i, "score": float(p[i])})
            pred.append({"video": v["video"], "events": evs})
        mAPs, _ = compute_mAPs(truth, pred, tolerances=[0, 1, 2, 4])
        return {t: float(mAPs[i]) * 100 for i, t in enumerate([0, 1, 2, 4])}

    out_dir = os.path.join(ROOT, "outputs", "action", run_name); os.makedirs(out_dir, exist_ok=True)
    mpath = os.path.join(out_dir, "metrics.csv"); open(mpath, "w").write("epoch,loss,mAP@0,mAP@1,mAP@2,mAP@4\n")
    best = -1.0
    for ep in range(epochs):
        model.train(); prng = random.Random(1000 + ep); order = tr_items[:]; prng.shuffle(order)
        rl, t0, nb = 0.0, time.time(), 0
        for b0 in range(0, (len(order) // batch_size) * batch_size, batch_size):
            chunk = order[b0:b0 + batch_size]
            x = torch.stack([torch.from_numpy(feats(v["video"])) for v, _ in chunk]).to(dev)  # [B,N,768]
            y = torch.stack([torch.from_numpy(make_label(v, t)) for v, t in chunk]).to(dev)    # [B,N]
            q = torch.stack([qvec[t] for _, t in chunk])
            m = torch.ones(x.shape[0], x.shape[1], 1, device=dev)
            outs = model(x, q, m)                              # [S,B,N,2]
            loss = sum(F.cross_entropy(outs[s].reshape(-1, 2), y.reshape(-1), weight=ce_w)
                       for s in range(outs.shape[0]))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            rl += float(loss.detach()); nb += 1
        ap = evaluate()
        print(f"[ep{ep}] loss {rl/max(nb,1):.3f} | mAP@0/1/2/4 = {[round(ap[t],2) for t in (0,1,2,4)]} "
              f"| {len(order)/(time.time()-t0):.0f} it/s", flush=True)
        open(mpath, "a").write(f"{ep},{rl/max(nb,1):.3f},{ap[0]:.2f},{ap[1]:.2f},{ap[2]:.2f},{ap[4]:.2f}\n")
        if ap[2] > best:
            best = ap[2]; torch.save({"model": model.state_dict(), "ap": ap, "lang": lang},
                                     os.path.join(out_dir, "best.pt"))
    print(f"[done] {run_name} best mAP@2 {best:.2f}", flush=True)
    return best


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_lang", action="store_true")
    ap.add_argument("--run_name", default="mstcn_lang")
    ap.add_argument("--epochs", type=int, default=50)
    a = ap.parse_args()
    run(lang=not a.no_lang, epochs=a.epochs, run_name=a.run_name)
