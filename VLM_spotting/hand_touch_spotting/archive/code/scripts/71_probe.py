"""Decisive interpretability: is CONTACT linearly decodable from the compressed per-frame
token? Build per-frame tokens (N,4096) from the connector, label frames contact (within
+/-TOL of a GT event) vs non-contact (>GAP from all GTs), fit a linear probe on TRAIN
videos, evaluate AUC/acc on VAL videos. High AUC => the idx-optimized tokens DO carry
contact semantics (even though not word-like / not a clean spatial focus). Compares the
frozen-compression connector (M1) vs the strongest SFT-LoRA connector."""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/home/chang_noroot/data2/hf_cache")
_COMMON = "/home/chang_noroot/data2/huyanh/Workspace/hand_touch_detection"
LAB = os.path.join(_COMMON, "data", "HOI4D-v3")
GRID = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/grid_even"
TOL, GAP = 2, 15                                                   # contact within +/-2, noncontact >15


def build_xy(fc, W, vids, dev, per_vid=None):
    X, y = [], []
    with torch.no_grad():
        for v in vids:
            k = v["video"].replace("/", "__")
            g = np.load(os.path.join(GRID, k + ".npy"))
            N = min(v["num_frames"], 2 * g.shape[0])
            tok = fc(torch.from_numpy(g), torch.zeros(1, W.d_llm), N).float().cpu()   # (N, d)
            gts = sorted(set(e["frame"] for e in v["events"] if e["frame"] < N))
            if not gts:
                continue
            con = [f for f in range(N) if any(abs(f - t) <= TOL for t in gts)]
            noncon = [f for f in range(N) if all(abs(f - t) > GAP for t in gts)]
            random.Random(k.__hash__() & 0xffff).shuffle(noncon)
            noncon = noncon[:len(con)]                             # balance per video
            for f in con:
                X.append(tok[f]); y.append(1)
            for f in noncon:
                X.append(tok[f]); y.append(0)
    return torch.stack(X), torch.tensor(y, dtype=torch.float32)


def fit_probe(Xtr, ytr, dev, epochs=300, wd=1e-3):
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xn = ((Xtr - mu) / sd).to(dev)
    w = torch.zeros(Xtr.shape[1], 1, device=dev, requires_grad=True)
    b = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05, weight_decay=wd)
    yt = ytr.to(dev).unsqueeze(1)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xn @ w + b, yt)
        loss.backward(); opt.step()
    return (w.detach(), b.detach(), mu.to(dev), sd.to(dev))


def evaluate(probe, X, y, dev):
    w, b, mu, sd = probe
    with torch.no_grad():
        s = (((X.to(dev) - mu) / sd) @ w + b).squeeze(1).cpu()
    yb = y.bool()
    # AUC via rank statistic (Mann-Whitney)
    order = s.argsort()
    ranks = torch.zeros_like(s); ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.float32)
    n1 = int(yb.sum()); n0 = len(y) - n1
    auc = (ranks[yb].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    acc = ((s > 0).float() == y).float().mean()
    return float(auc), float(acc), n1, n0


def main():
    dev = "cuda"; torch.set_num_threads(8)
    from models.wrapper import QwenWrapper
    from models.frame_compress import FrameCompress
    W = QwenWrapper(device=dev, dtype=torch.bfloat16)

    def load_fc(ckpt):
        fc = FrameCompress(768, 8, W.d_llm, n_heads=4, stain=False, use_text=False).to(dev, torch.float32)
        fc.set_target_rms_from(W.embed_tokens.weight)
        fc.load_state_dict(torch.load(os.path.join(ROOT, ckpt), weights_only=False)["fc"]); fc.eval()
        return fc

    def have(v):
        return os.path.exists(os.path.join(GRID, v["video"].replace("/", "__") + ".npy")) and len(v["events"]) >= 1
    tr = [v for v in json.load(open(os.path.join(LAB, "train.json"))) if have(v)]
    va = [v for v in json.load(open(os.path.join(LAB, "val.json"))) if have(v)]
    random.Random(0).shuffle(tr); tr = tr[:250]; va = va[:120]
    print(f"[probe] train vids={len(tr)} val vids={len(va)} (TOL=+/-{TOL}, GAP>{GAP})", flush=True)

    for name, ckpt in [("M1 frozen-compress", "outputs/idx_compress/comp_notext/best.pt"),
                       ("SFT-LoRA (strongest)", "outputs/idx_compress/sft_lora_v2/best.pt")]:
        fc = load_fc(ckpt)
        Xtr, ytr = build_xy(fc, W, tr, dev)
        Xva, yva = build_xy(fc, W, va, dev)
        probe = fit_probe(Xtr, ytr, dev)
        auc, acc, n1, n0 = evaluate(probe, Xva, yva, dev)
        # random-token control: shuffle labels baseline is AUC~0.5 by construction
        print(f"\n=== {name} ===\n  linear-probe VAL  AUC={auc:.3f}  acc={acc:.3f}  "
              f"(val contact={n1} noncontact={n0}; train={len(ytr)})", flush=True)
        del fc, Xtr, Xva; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
