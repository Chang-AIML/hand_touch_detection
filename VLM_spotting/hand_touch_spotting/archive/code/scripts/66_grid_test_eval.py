"""Eval a trained GridRefine checkpoint on the TEST split (dense + no/NMS/soft-NMS),
using the extracted test window grids. Headline number for the token-pool method."""
from __future__ import annotations

import argparse
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
sys.path.insert(0, _COMMON); sys.path.insert(0, os.path.join(_COMMON, "methods/spot_head"))
LAB = os.path.join(_COMMON, "data", "HOI4D-v3")
CACHE = os.path.join(ROOT, "outputs/action/cache")
GRID = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/grid_win"
from refine_grid import GridRefine                                # noqa: E402
from common.eval import non_maximum_supression                    # noqa: E402
import eval_nms as en                                             # noqa: E402

dev = "cuda"; types = ("touch", "untouch"); tid = {"touch": 0, "untouch": 1}
en.TOLS = [0, 1, 2]
tq = torch.load(os.path.join(CACHE, "type_queries.pt"), weights_only=False)
qv = {t: tq[t].to(dev) for t in types}
gc = {}


def gload(k):
    if k not in gc:
        d = np.load(os.path.join(GRID, k + ".npz"))
        fr = d["frames"].astype(np.int64); grids = d["grids"]
        pos = np.full(int(fr.max()) + 1, -1, np.int64); pos[fr] = np.arange(len(fr))
        gc[k] = (pos, grids)
    return gc[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True); ap.add_argument("--pool", required=True)
    ap.add_argument("--W", type=int, default=24); ap.add_argument("--out_E", type=int, default=12)
    a = ap.parse_args()
    W, out_E, L = a.W, a.out_E, 2 * a.W + 1
    have = set(os.path.splitext(f)[0] for f in os.listdir(GRID))
    model = GridRefine(a.pool, 128).to(dev)
    model.load_state_dict(torch.load(os.path.join(ROOT, "outputs/spot", a.run_name, "best.pt"),
                                     weights_only=False)["model"]); model.eval()
    coarse = torch.load(os.path.join(CACHE, "idx_multi_test_0.pt"), weights_only=False)["coarse"]
    vids = [v for v in json.load(open(os.path.join(LAB, "test.json")))
            if v["video"].replace("/", "__") in have]
    truth = [{"video": v["video"], "num_frames": v["num_frames"],
              "events": [{"label": e["label"], "frame": e["frame"]} for e in v["events"]]} for v in vids]

    def win(centers, ks, ts):
        B = len(centers); g = np.zeros((B, L, 576, 768), np.float16); m = np.zeros((B, L), np.float32)
        for i, (c, k) in enumerate(zip(centers, ks)):
            pos, grids = gload(k); fr = np.arange(c - W, c + W + 1)
            valid = (fr >= 0) & (fr < len(pos)); rows = np.where(valid, pos[np.clip(fr, 0, len(pos) - 1)], -1)
            ok = rows >= 0; g[i, ok] = grids[rows[ok]]; m[i, ok] = 1
        return (torch.from_numpy(g).to(dev).float(), torch.from_numpy(m).unsqueeze(1).to(dev),
                torch.tensor([tid[x] for x in ts]).to(dev))

    flat = [(v["video"], v["video"].replace("/", "__"), t, int(p))
            for v in vids for t in types for p, _ in coarse.get((v["video"].replace("/", "__"), t), [])]
    det = {v["video"]: [] for v in vids}
    with torch.no_grad():
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
    pl = [{"video": v["video"], "events": det[v["video"]]} for v in vids]
    print(f"=== {a.run_name} (pool={a.pool}) TEST ===")
    for name, pv in (("without NMS", pl), ("NMS(w=1)", non_maximum_supression(pl, 1)),
                     ("soft-NMS", en.soft_nms(pl, 4, 0.5))):
        m = en.maps_quiet(truth, pv)
        print(f"  {name:<12} @0/1/2 = {[round(x,2) for x in m]}")


if __name__ == "__main__":
    main()
