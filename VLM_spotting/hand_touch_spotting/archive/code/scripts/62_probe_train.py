"""PROBE: is the frame-level contact signal RECOVERABLE from the token grid?
Task = classify the EXACT contact frame (off==0) vs an immediate neighbor (|off|==1),
from a single frame's (576,768) grid. Compare pooling strategies (same MLP head):
  A. mean-pool (what feat_interleave does)   -> expected ~50% (signal averaged away)
  B. learned-query attention-pool             -> can it SELECT contact tokens?
  C. type-conditioned attention-pool
  D. 4-head attention-pool
Split by video. If B/C/D >> A and >50%, the signal is in the tokens -> build the pipeline.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
dev = "cuda"


class Pool(nn.Module):
    def __init__(self, kind, d=768, heads=1):
        super().__init__()
        self.kind = kind; self.heads = heads
        if kind in ("attn", "type", "multi"):
            self.q = nn.Parameter(torch.randn(heads, d) * 0.02)
            self.kp = nn.Linear(d, d); self.vp = nn.Linear(d, d)
            if kind == "type":
                self.temb = nn.Embedding(2, d)
        self.head = nn.Sequential(nn.LayerNorm(d * (heads if kind == "multi" else 1)),
                                  nn.Linear(d * (heads if kind == "multi" else 1), 128),
                                  nn.GELU(), nn.Linear(128, 2))

    def forward(self, x, t):                                       # x[B,576,768] t[B]
        if self.kind == "mean":
            z = x.mean(1)
        else:
            k = self.kp(x); v = self.vp(x)                        # [B,576,d]
            q = self.q.unsqueeze(0).expand(x.size(0), -1, -1).clone()  # [B,H,d]
            if self.kind == "type":
                q = q + self.temb(t).unsqueeze(1)
            a = torch.softmax((q @ k.transpose(1, 2)) / (x.size(-1) ** 0.5), -1)  # [B,H,576]
            pooled = a @ v                                        # [B,H,d]
            z = pooled.reshape(x.size(0), -1) if self.kind == "multi" else pooled.mean(1)
        return self.head(z)


def run(kind, Xtr, ttr, ytr, Xva, tva, yva, heads=1, epochs=40, bs=256, lr=1e-3):
    torch.manual_seed(0)
    m = Pool(kind, heads=heads).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    n = len(Xtr); best = 0.0
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n)
        for b in range(0, n, bs):
            idx = perm[b:b + bs]
            xb = torch.from_numpy(Xtr[idx.numpy()]).float().to(dev)
            tb = torch.from_numpy(ttr[idx.numpy()]).long().to(dev)
            yb = torch.from_numpy(ytr[idx.numpy()]).long().to(dev)
            loss = F.cross_entropy(m(xb, tb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pr = []
            for b in range(0, len(Xva), 512):
                xb = torch.from_numpy(Xva[b:b + 512]).float().to(dev)
                tb = torch.from_numpy(tva[b:b + 512]).long().to(dev)
                pr.append(m(xb, tb).argmax(1).cpu().numpy())
            pred = np.concatenate(pr)
            # balanced accuracy (mean of per-class recall) -> chance 0.5
            recs = [(pred[yva == c] == c).mean() for c in (0, 1) if (yva == c).any()]
            acc = float(np.mean(recs))
        best = max(best, acc)
    return best


def task(Xa, offa, typa, vidixa, ya, name):
    vids = np.unique(vidixa); nval = max(1, len(vids) // 4)
    val_v = set(vids[-nval:].tolist())
    tr = np.array([v not in val_v for v in vidixa]); va = ~tr
    args = (Xa[tr], typa[tr], ya[tr], Xa[va], typa[va], ya[va])
    print(f"\n[{name}]  {len(Xa)} frames | pos {int(ya.sum())} neg {int((1-ya).sum())} | "
          f"train {tr.sum()} val {va.sum()} | balanced-acc chance 0.500", flush=True)
    print(f"  A mean-pool       bal-acc = {run('mean', *args):.3f}   <- feat_interleave pooling", flush=True)
    print(f"  B attn(1q)        bal-acc = {run('attn', *args):.3f}", flush=True)
    print(f"  C type-attn       bal-acc = {run('type', *args):.3f}", flush=True)
    print(f"  D 4-head attn     bal-acc = {run('multi', *args, heads=4):.3f}", flush=True)


def main():
    d = np.load(os.path.join(ROOT, "outputs/probe/grids.npz"))
    X, off, typ, vidix = d["X"], d["off"], d["typ"], d["vidix"]

    # TARGET: exact (off==0) vs immediate neighbor (|off|==1)
    m = np.abs(off) <= 1
    task(X[m], off[m], typ[m], vidix[m], (off[m] == 0).astype(np.int64), "TARGET exact vs ±1")
    # CONTROL 1 (coarse temporal): exact (0) vs far (|off|==3)
    m = np.isin(off, [0, -3, 3])
    task(X[m], off[m], typ[m], vidix[m], (off[m] == 0).astype(np.int64), "CTRL coarse exact vs ±3")
    # CONTROL 2 (semantic sanity, must be easy): touch vs untouch (all frames)
    task(X, off, typ, vidix, typ.astype(np.int64), "CTRL semantic touch vs untouch")


if __name__ == "__main__":
    main()
