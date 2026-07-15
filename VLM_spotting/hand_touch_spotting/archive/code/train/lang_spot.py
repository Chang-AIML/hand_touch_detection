"""Flexible spotting-head trainer for the overnight method search.

Feature-agnostic (F1/F2/F3/...) via --feat_dir. clip-100 windows + random-crop aug
(matches the SOTA pipeline). Two modes:
  multiclass : per-frame {bg,touch,untouch}  (== their MS-TCN, the 69.3 setup)
  query      : per-type binary, FiLM(type-query) conditions the MS-TCN (language)
E2E fg-weighted CE across stages. Eval = sliding-window full-video -> mAP@0/1/2.
Backbone: MS-TCN (3-stage). Optional idx-prior channel (--idx_prior).
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
LAB = os.path.join(_COMMON, "data", "HOI4D-v3")
from common.score import compute_mAPs                          # noqa: E402


class SingleStageTCN(nn.Module):
    class DRL(nn.Module):
        def __init__(self, dil, c):
            super().__init__()
            self.cd = nn.Conv1d(c, c, 3, padding=dil, dilation=dil)
            self.c1 = nn.Conv1d(c, c, 1); self.do = nn.Dropout()

        def forward(self, x, m):
            return (x + self.do(self.c1(F.relu(self.cd(x))))) * m[:, 0:1, :]

    def __init__(self, in_dim, hid, out_dim, n_layers):
        super().__init__()
        self.c1 = nn.Conv1d(in_dim, hid, 1)
        self.layers = nn.ModuleList([SingleStageTCN.DRL(2 ** i, hid) for i in range(n_layers)])
        self.co = nn.Conv1d(hid, out_dim, 1)

    def forward(self, x, m):                                   # x[B,L,C] m[B,L,1]
        m = m.permute(0, 2, 1); x = self.c1(x.permute(0, 2, 1))
        for l in self.layers:
            x = l(x, m)
        return (self.co(x) * m[:, 0:1, :]).permute(0, 2, 1)


class SpotHead(nn.Module):
    def __init__(self, feat_dim, num_classes, num_stages=3, hid=256, n_layers=5,
                 d_llm=4096, use_film=False):
        super().__init__()
        self.use_film = use_film
        self.stage1 = SingleStageTCN(feat_dim, hid, num_classes, n_layers)
        self.stages = nn.ModuleList([SingleStageTCN(num_classes, hid, num_classes, n_layers)
                                     for _ in range(num_stages - 1)])
        if use_film:
            self.film = nn.Sequential(nn.Linear(d_llm, hid), nn.GELU(), nn.Linear(hid, 2 * feat_dim))
            nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)

    def forward(self, x, m, q=None):
        if self.use_film and q is not None:
            g, b = self.film(q).chunk(2, dim=-1)
            x = (1 + g).unsqueeze(1) * x + b.unsqueeze(1)
        out = self.stage1(x, m); outs = [out]
        for s in self.stages:
            out = s(F.softmax(out, dim=2) * m, m); outs.append(out)
        return torch.stack(outs)


def run(feat_dir, mode="multiclass", backbone="mstcn", num_stages=3, dilate_lbl=4,
        fg_weight=5.0, clip_len=100, pad_len=5, epochs=50, batch_size=16, lr=1e-3,
        run_name="lang_spot", seed=0, d_llm=4096):
    torch.manual_seed(seed); dev = "cuda"; types = ("touch", "untouch")
    query_cond = (mode == "query")
    tq = torch.load(os.path.join(ROOT, "outputs/action/cache/type_queries.pt"), weights_only=False)
    qvec = {t: tq[t].to(dev) for t in types}

    def fpath(video):
        return os.path.join(feat_dir, video.replace("/", "__") + ".npy")

    fc = {}

    def feats(video):
        if video not in fc:
            fc[video] = np.load(fpath(video)).astype(np.float32)
        return fc[video]

    def load_vids(split):
        return [v for v in json.load(open(os.path.join(LAB, f"{split}.json")))
                if os.path.exists(fpath(v["video"]))]

    tr_vids, val_vids = load_vids("train"), load_vids("val")
    feat_dim = feats(tr_vids[0]["video"]).shape[-1]
    nc = 2 if query_cond else 3                                # bg,event | bg,touch,untouch
    cls_idx = {"touch": 1, "untouch": 2}
    # training items
    if query_cond:
        tr_items = [(v, t) for v in tr_vids for t in types if any(e["label"] == t for e in v["events"])]
    else:
        tr_items = [(v, None) for v in tr_vids]
    print(f"[cfg] {run_name} feat={os.path.basename(feat_dir)}({feat_dim}) mode={mode} "
          f"stages={num_stages} clip={clip_len} | train {len(tr_items)} val {len(val_vids)}", flush=True)

    model = SpotHead(feat_dim, nc, num_stages, use_film=query_cond, d_llm=d_llm).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR
    ds_len = 1000000 // clip_len                              # ~10000 samples/epoch (matches SOTA)
    spe = ds_len // batch_size                                # steps per epoch (matches the loop!)
    sched = ChainedScheduler([LinearLR(opt, 0.01, 1.0, 3 * spe),
                              CosineAnnealingLR(opt, spe * (epochs - 3))])
    ce_w = torch.tensor([1.0] + [fg_weight] * (nc - 1), device=dev)

    def clip_sample(v, t):
        f = feats(v["video"]); N = v["num_frames"]
        bi = random.randint(-pad_len, max(-pad_len, N - clip_len - 1 + pad_len)) if N > clip_len + 1 else pad_len
        y = np.zeros(clip_len, dtype=np.int64)
        for e in v["events"]:
            if query_cond and e["label"] != t:
                continue
            li = e["frame"] - bi
            if -dilate_lbl <= li < clip_len + dilate_lbl:
                lab = 1 if query_cond else cls_idx[e["label"]]
                for i in range(max(0, li - dilate_lbl), min(clip_len, li + dilate_lbl + 1)):
                    y[i] = lab
        pb = max(0, bi); fe = f[pb:pb + clip_len]
        msk = np.ones(clip_len, dtype=np.float32); msk[fe.shape[0]:] = 0
        if fe.shape[0] < clip_len:
            ps = 0 if bi > 0 else -bi
            fe = np.pad(fe, ((ps, clip_len - fe.shape[0] - ps), (0, 0)))
        return fe, y, msk

    @torch.no_grad()
    def evaluate():
        model.eval()
        truth = [{"video": v["video"], "num_frames": v["num_frames"],
                  "events": [{"label": e["label"], "frame": e["frame"]} for e in v["events"]]}
                 for v in val_vids]
        pred = []
        for v in val_vids:
            f = feats(v["video"]); N = f.shape[0]
            fp = np.pad(f, ((pad_len, pad_len), (0, 0)))       # FULL-video eval (MS-TCN needs full context)
            x = torch.from_numpy(fp).float().unsqueeze(0).to(dev)
            m = torch.ones(1, fp.shape[0], 1, device=dev)
            evs = []
            qs = types if query_cond else [None]
            for t in qs:
                q = qvec[t].unsqueeze(0) if query_cond else None
                sc = F.softmax(model(x, m, q)[-1][0], dim=-1).cpu().numpy()[pad_len:-pad_len]
                if query_cond:
                    for i in range(N):
                        if sc[i, 1] >= 0.01:
                            evs.append({"label": t, "frame": i, "score": float(sc[i, 1])})
                else:
                    for i in range(N):
                        for c, lab in ((1, "touch"), (2, "untouch")):
                            if sc[i, c] >= 0.01:
                                evs.append({"label": lab, "frame": i, "score": float(sc[i, c])})
            pred.append({"video": v["video"], "events": evs})
        mAPs, _ = compute_mAPs(truth, pred, tolerances=[0, 1, 2, 4])
        return {t: float(mAPs[i]) * 100 for i, t in enumerate([0, 1, 2, 4])}

    out_dir = os.path.join(ROOT, "outputs", "spot", run_name); os.makedirs(out_dir, exist_ok=True)
    mp = os.path.join(out_dir, "metrics.csv"); open(mp, "w").write("epoch,loss,mAP@0,mAP@1,mAP@2,mAP@4\n")
    best = -1.0
    for ep in range(epochs):
        model.train(); rl, t0, nb = 0.0, time.time(), 0
        random.seed(seed + ep * 97)
        for _ in range(ds_len // batch_size):
            chunk = [random.choice(tr_items) for _ in range(batch_size)]
            xb, yb, mb, qb = [], [], [], []
            for v, t in chunk:
                fe, y, msk = clip_sample(v, t)
                xb.append(fe); yb.append(y); mb.append(msk)
                if query_cond:
                    qb.append(qvec[t])
            x = torch.from_numpy(np.stack(xb)).float().to(dev)
            y = torch.from_numpy(np.stack(yb)).to(dev)
            m = torch.from_numpy(np.stack(mb)).float().unsqueeze(-1).to(dev)
            q = torch.stack(qb) if query_cond else None
            outs = model(x, m, q)
            loss = sum(F.cross_entropy((outs[s] * m).reshape(-1, nc), y.reshape(-1), weight=ce_w)
                       for s in range(outs.shape[0]))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            rl += float(loss.detach()); nb += 1
        ap = evaluate()
        print(f"[ep{ep}] loss {rl/max(nb,1):.3f} | mAP@0/1/2/4 = {[round(ap[t],2) for t in (0,1,2,4)]}", flush=True)
        open(mp, "a").write(f"{ep},{rl/max(nb,1):.3f},{ap[0]:.2f},{ap[1]:.2f},{ap[2]:.2f},{ap[4]:.2f}\n")
        if ap[2] > best:
            best = ap[2]; torch.save({"model": model.state_dict(), "ap": ap}, os.path.join(out_dir, "best.pt"))
    print(f"[done] {run_name} best mAP@2 {best:.2f} @0 {torch.load(os.path.join(out_dir,'best.pt'),weights_only=False)['ap'][0]:.2f}", flush=True)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--mode", default="multiclass", choices=["multiclass", "query"])
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--num_stages", type=int, default=3)
    ap.add_argument("--dilate", type=int, default=0, help="label radius (0=sharp, matches SOTA)")
    a = ap.parse_args()
    run(feat_dir=a.feat_dir, mode=a.mode, run_name=a.run_name, epochs=a.epochs,
        num_stages=a.num_stages, dilate_lbl=a.dilate)
