"""Eval an idx checkpoint at extra tolerances {0,1,2,4,8,16} + build a per-event
localization ERROR distribution (each GT event -> |gt - nearest same-type prediction|).
Saves a histogram+CDF plot."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/home/chang_noroot/data2/hf_cache")
_COMMON = "/home/chang_noroot/data2/huyanh/Workspace/hand_touch_detection"
sys.path.insert(0, _COMMON)
FEAT = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/feat_interleave"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "outputs/idx/idx_multi/best.pt"))
    ap.add_argument("--mrope", action="store_true")
    ap.add_argument("--split", default="val")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs/idx/idx_multi/error_dist.png"))
    args = ap.parse_args()

    from models.wrapper import QwenWrapper
    from models.vjepa_adaptor import VJEPAAdaptor
    from models.idx_localizer import IdxLocalizer
    from data.questions import GENERIC_Q
    from common.score import compute_mAPs

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]; rank = cfg.get("lora_rank", 16)
    W = QwenWrapper(device="cuda", dtype=torch.bfloat16)
    ad = VJEPAAdaptor(768, W.d_llm, hidden=2048, n_layers=2).to("cuda", torch.bfloat16)
    ad.set_target_rms_from(W.embed_tokens.weight)     # NOT in state_dict -> must restore!
    ad.load_state_dict(ck["adaptor"]); ad.eval()
    if cfg.get("lora") and "lora" in ck:
        lp = W.add_lora(rank=rank, alpha=2 * rank, n_layers=W.n_layers, target="all")
        for p, s in zip(lp, ck["lora"]):
            p.data.copy_(s.to(p.device, p.dtype))
    loz = IdxLocalizer(W, ad, use_idx=cfg.get("use_idx", True), use_anchor=True,
                       anchor_stride=5, anchor_max_side=252, use_mrope=args.mrope,
                       fps=15, max_frames=320, grad_checkpoint=False)

    vids = json.load(open(os.path.join(ROOT, "data", "annotations", f"{args.split}.json")))
    vids = [v for v in vids if os.path.exists(os.path.join(FEAT, v["video_id"] + ".npy"))]
    types = ("touch", "untouch")
    items = [(v["video_id"], t) for v in vids for t in types
             if any(e["type"] == t for e in v["events"])]
    feat_cache = {}

    def feats_of(vid):
        if vid not in feat_cache:
            feat_cache[vid] = np.load(os.path.join(FEAT, vid + ".npy")).astype(np.float32)
        return feat_cache[vid]

    truth = {v["video_id"]: [{"label": e["type"], "frame": e["frame"]} for e in v["events"]]
             for v in vids}
    pr = {v["video_id"]: [] for v in vids}
    preds_by = {}                    # (vid,type) -> [frames]
    for b0 in range(0, len(items), args.batch_size):
        chunk = items[b0:b0 + args.batch_size]
        samples = []
        for vid, typ in chunk:
            f = feats_of(vid); N = f.shape[0]
            samples.append({"feats": torch.from_numpy(f), "question": GENERIC_Q[typ],
                            "event_frames": None, "gt": -1, "fps": 15, "type": typ,
                            "video_id": vid, "num_frames": N, "full_num_frames": N,
                            "anchor_start_sec": 0, "anchor_num_secs": N // 15})
        outs = loz.predict_multievent_batch(samples)
        for (vid, typ), (frames, scores) in zip(chunk, outs):
            N = feats_of(vid).shape[0]
            fl = [int(f) for f in frames if 0 <= f < N]
            preds_by[(vid, typ)] = fl
            for f, sc in zip(frames, scores):
                if 0 <= f < N:
                    pr[vid].append({"label": typ, "frame": int(f), "score": float(sc)})
        if (b0 // args.batch_size) % 10 == 0:
            print(f"  {b0}/{len(items)}", flush=True)

    order = sorted(truth)
    tol = [0, 1, 2, 4, 8, 16]
    tl = [{"video": v, "events": truth[v]} for v in order]
    pl = [{"video": v, "events": pr[v]} for v in order]
    mAPs, _ = compute_mAPs(tl, pl, tolerances=tol)
    print("\n=== mAP (touch+untouch avg) ===")
    for t, m in zip(tol, mAPs):
        print(f"  mAP@{t:<2} = {float(m)*100:.2f}")

    # per-event error: each GT -> |gt - nearest same-type prediction| (999 if none)
    errs = {"touch": [], "untouch": []}
    for v in vids:
        vid = v["video_id"]
        for e in v["events"]:
            typ = e["type"]; gt = e["frame"]
            ps = preds_by.get((vid, typ), [])
            errs[typ].append(min((abs(gt - p) for p in ps), default=999))
    allm = np.array(errs["touch"] + errs["untouch"])
    for k in ("touch", "untouch"):
        a = np.array(errs[k]); hit = a < 999
        print(f"\n[{k}] n={len(a)} miss(no-pred)={100*(~hit).mean():.1f}% "
              f"median={np.median(a[hit]):.1f} mean={a[hit].mean():.1f}")
        for kk in (0, 1, 2, 4, 8, 16):
            print(f"    within {kk:>2}f: {100*(a <= kk).mean():.1f}%")

    # plot: histogram (0..30) + CDF, per type
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (axh, axc) = plt.subplots(1, 2, figsize=(13, 4.5))
    bins = np.arange(0, 32)
    for k, c in [("touch", "#d1495b"), ("untouch", "#2e86ab")]:
        a = np.array(errs[k]); a = a[a < 999]
        axh.hist(np.clip(a, 0, 31), bins=bins, alpha=0.6, label=f"{k} (n={len(a)})", color=c)
        xs = np.arange(0, 31)
        cdf = [100 * (a <= x).mean() for x in xs]
        axc.plot(xs, cdf, label=k, color=c, lw=2)
    axh.set_xlabel("localization error |pred - GT| (frames)"); axh.set_ylabel("count")
    axh.set_title("Error histogram (nearest pred per GT event)"); axh.legend()
    axc.set_xlabel("tolerance (frames)"); axc.set_ylabel("% GT events within tolerance")
    axc.set_title("Cumulative (recall vs tolerance)"); axc.grid(alpha=0.3); axc.legend()
    for x in (0, 1, 2, 4, 8, 16):
        axc.axvline(x, color="gray", ls=":", lw=0.6)
    fig.suptitle(f"idx-LLM localization error — {os.path.basename(os.path.dirname(args.ckpt))} "
                 f"(mrope={args.mrope}, {args.split})")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"\n[plot] {args.out}")


if __name__ == "__main__":
    main()
