"""GENERALITY test of the HOI4D-trained 8-query compression connector, via linear probe on
UNSEEN TACO. Freeze the connector, extract per-frame compressed tokens on TACO, probe for
'is this an event frame' (touch/untouch within +/-2 vs non-event >15). Matrix:
  HOI4D->HOI4D  (reference, expect ~0.89)
  TACO ->TACO   (does the connector encode contact on unseen TACO at all?)
  HOI4D->TACO   (STRICT: is the SAME linear contact direction present in TACO tokens?)
  TACO ->HOI4D  (reverse transfer)
+ shuffled-label control (must be ~0.50) as an adversarial sanity check.
Run for M1 (frozen-compress) and SFT-LoRA connectors. Reuses the scripts/71 probe."""
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
TM = "/home/chang_noroot/data2/huyanh/Workspace/dataset/hoi4d/touchmoment"
GRID = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/grid_even"
TOL, GAP = 2, 15
N_TR, N_TE = 300, 225                                             # videos per probe-train / probe-test


def split_ids(kind):
    """kind in {'hoi_tr','hoi_te','taco_tr','taco_te'}: list of (video, events) w/ features."""
    src = {"hoi_tr": "train", "hoi_te": "val", "taco_tr": "train", "taco_te": "test"}[kind]
    isH = kind.startswith("hoi")
    out = []
    for v in json.load(open(os.path.join(TM, f"{src}.json"))):
        vid = v["video"]
        if (vid.startswith("H")) != isH:
            continue
        if not os.path.exists(os.path.join(GRID, vid.replace("/", "__") + ".npy")):
            continue
        if not v["events"]:
            continue
        out.append(v)
    random.Random(0).shuffle(out)
    return out


def build_xy(fc, W, vids, dev):
    X, y = [], []
    with torch.no_grad():
        for v in vids:
            k = v["video"].replace("/", "__")
            g = np.load(os.path.join(GRID, k + ".npy"))
            N = min(v["num_frames"], 2 * g.shape[0])
            tok = fc(torch.from_numpy(g), torch.zeros(1, W.d_llm), N).float().cpu()   # (N,d)
            gts = sorted(set(e["frame"] for e in v["events"] if e["frame"] < N))
            if not gts:
                continue
            con = [f for f in range(N) if any(abs(f - t) <= TOL for t in gts)]
            non = [f for f in range(N) if all(abs(f - t) > GAP for t in gts)]
            random.Random(hash(k) & 0xffff).shuffle(non); non = non[:len(con)]        # balance/video
            for f in con:
                X.append(tok[f]); y.append(1)
            for f in non:
                X.append(tok[f]); y.append(0)
    return torch.stack(X), torch.tensor(y, dtype=torch.float32)


def fit_probe(Xtr, ytr, dev, epochs=200, wd=3.0):
    # STRONG L2 (wd) so a 4096-dim probe can't memorize noise -> shuffled control ~0.5.
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xn = ((Xtr - mu) / sd).to(dev)
    w = torch.zeros(Xtr.shape[1], 1, device=dev, requires_grad=True)
    b = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05, weight_decay=wd); yt = ytr.to(dev).unsqueeze(1)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Xn @ w + b, yt)
        loss.backward(); opt.step()
    return (w.detach(), b.detach(), mu.to(dev), sd.to(dev))


def evaluate(probe, X, y, dev):
    w, b, mu, sd = probe
    with torch.no_grad():
        s = (((X.to(dev) - mu) / sd) @ w + b).squeeze(1).cpu()
    yb = y.bool(); order = s.argsort()
    ranks = torch.zeros_like(s); ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.float32)
    n1 = int(yb.sum()); n0 = len(y) - n1
    auc = (ranks[yb].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    acc = ((s > 0).float() == y).float().mean()
    return float(auc), float(acc), n1, n0


CACHE = os.path.join(ROOT, "plot", "probe_taco_xy")               # per-connector token cache


def main():
    dev = "cuda"; torch.set_num_threads(8); torch.manual_seed(0)
    from models.wrapper import QwenWrapper
    from models.frame_compress import FrameCompress
    os.makedirs(CACHE, exist_ok=True)
    replot = "--replot" in sys.argv
    W = None if replot and all(os.path.exists(os.path.join(CACHE, f"{c}_{k}.npz"))
                               for c in ("M1", "SFT") for k in ("hoi_tr", "hoi_te", "taco_tr", "taco_te")) else \
        QwenWrapper(device=dev, dtype=torch.bfloat16)

    def load_fc(ckpt):
        fc = FrameCompress(768, 8, W.d_llm, n_heads=4, stain=False, use_text=False).to(dev, torch.float32)
        fc.set_target_rms_from(W.embed_tokens.weight)
        fc.load_state_dict(torch.load(os.path.join(ROOT, ckpt), weights_only=False)["fc"]); fc.eval()
        return fc

    sets = {k: split_ids(k) for k in ("hoi_tr", "hoi_te", "taco_tr", "taco_te")}
    sets["hoi_tr"] = sets["hoi_tr"][:N_TR]; sets["taco_tr"] = sets["taco_tr"][:N_TR]
    sets["hoi_te"] = sets["hoi_te"][:N_TE]; sets["taco_te"] = sets["taco_te"][:N_TE]
    print(f"[probe] videos: hoi_tr={len(sets['hoi_tr'])} hoi_te={len(sets['hoi_te'])} "
          f"taco_tr={len(sets['taco_tr'])} taco_te={len(sets['taco_te'])} (TOL={TOL} GAP={GAP}, wd strong)", flush=True)

    def get_xy(cshort, ckpt):
        fc = None
        out = {}
        for k in sets:
            cp = os.path.join(CACHE, f"{cshort}_{k}.npz")
            if os.path.exists(cp):
                z = np.load(cp); out[k] = (torch.from_numpy(z["X"]), torch.from_numpy(z["y"]))
            else:
                if fc is None:
                    fc = load_fc(ckpt)
                X, y = build_xy(fc, W, sets[k], dev)
                np.savez(cp, X=X.numpy(), y=y.numpy()); out[k] = (X, y)
        return out

    def perm_null(Xtr, ytr, Xte, yte, k=5):                       # mean AUC over k label shuffles (~0.5 if honest)
        aucs = []
        for s in range(k):
            g = torch.Generator().manual_seed(100 + s)
            ysh = ytr[torch.randperm(len(ytr), generator=g)]
            aucs.append(evaluate(fit_probe(Xtr, ysh, dev), Xte, yte, dev)[0])
        return float(np.mean(aucs)), float(np.std(aucs))

    for cshort, cname, ckpt in [("M1", "M1 frozen-compress", "outputs/idx_compress/comp_notext/best.pt"),
                                ("SFT", "SFT-LoRA", "outputs/idx_compress/sft_lora_v2/best.pt")]:
        XY = get_xy(cshort, ckpt)
        pH = fit_probe(*XY["hoi_tr"], dev); pT = fit_probe(*XY["taco_tr"], dev)
        print(f"\n=== {cname} ===", flush=True)
        for name, probe, teset in [
                ("HOI4D -> HOI4D  (reference)", pH, "hoi_te"),
                ("TACO  -> TACO   (within new)", pT, "taco_te"),
                ("HOI4D -> TACO   (STRICT transfer)", pH, "taco_te"),
                ("TACO  -> HOI4D  (reverse)", pT, "hoi_te")]:
            auc, acc, n1, n0 = evaluate(probe, *XY[teset], dev)
            print(f"  {name:<38} AUC={auc:.3f} acc={acc:.3f}  (test +{n1}/-{n0})", flush=True)
        m, s = perm_null(*XY["hoi_tr"], *XY["taco_te"])
        print(f"  {'permutation null (HOI4D-shuf -> TACO)':<38} AUC={m:.3f}±{s:.3f}  (should be ~0.50)", flush=True)


if __name__ == "__main__":
    main()
