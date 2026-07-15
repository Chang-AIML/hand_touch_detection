"""PRECISION-vs-tolerance figure for the BEST model (SFT-LoRA, 72.9 test), reconciled with
AP@2. Uses AP-style ONE-TO-ONE matching (sort preds by score, each matches nearest unmatched
same-label GT; leftover preds = duplicate false-positives), so 'precision@tol' is honest
(dedup), unlike a naive nearest-GT count. Shows WHY AP@2=72.9 < precision or recall alone:
AP@2 = recall@2 x rank-interp-precision. sqrt x-axis magnifies low tolerance. Caches the raw
predictions so replotting needs no re-inference (pass --replot)."""
from __future__ import annotations

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
sys.path.insert(0, _COMMON); sys.path.insert(0, os.path.join(_COMMON, "methods/spot_head"))
LAB = os.path.join(_COMMON, "data", "HOI4D-v3")
GRID = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/grid_even"
import eval_nms as en                                             # noqa: E402
from data.questions import GENERIC_Q                              # noqa: E402

CKPT = "outputs/idx_compress/sft_lora_v2/best.pt"                 # the strongest model
PRED_CACHE = os.path.join(ROOT, "plot", "precision_preds.json")   # raw preds -> fast replot


def get_preds(W, vids, dev, batch_size=8):
    from models.vjepa_adaptor import VJEPAAdaptor
    from models.idx_localizer import IdxLocalizer
    from models.frame_compress import FrameCompress
    ck = torch.load(os.path.join(ROOT, CKPT), weights_only=False)
    ad = VJEPAAdaptor(768, W.d_llm, hidden=2048, n_layers=2).to(dev, torch.bfloat16)
    fc = FrameCompress(768, 8, W.d_llm, n_heads=4, stain=False, use_text=False).to(dev, torch.float32)
    fc.set_target_rms_from(W.embed_tokens.weight); fc.load_state_dict(ck["fc"]); fc.eval()
    lp = W.add_lora(rank=16, alpha=32, n_layers=W.n_layers, target="all")
    for p, s in zip(lp, ck["lora"]):
        p.data.copy_(s.to(p.device, p.dtype))
    loz = IdxLocalizer(W, ad, use_idx=True, use_anchor=True, anchor_stride=5, anchor_max_side=252,
                       use_mrope=False, fps=15, max_frames=320, grad_checkpoint=False, compress=fc)
    items = [(v, t) for v in vids for t in ("touch", "untouch") if any(e["label"] == t for e in v["events"])]
    fcache = {}

    def sample_of(v, t):
        k = v["video"].replace("/", "__")
        if k not in fcache:
            fcache[k] = np.load(os.path.join(GRID, k + ".npy"))
        g = fcache[k]; N = min(v["num_frames"], 2 * g.shape[0])
        return {"grid": torch.from_numpy(g), "question": GENERIC_Q[t], "type": t, "video_id": k,
                "fps": 15, "num_frames": N, "full_num_frames": N, "anchor_start_sec": 0,
                "anchor_num_secs": N // 15, "event_frames": None, "gt": -1}

    pr = {v["video"]: [] for v in vids}
    with torch.no_grad():
        for b0 in range(0, len(items), batch_size):
            ch = items[b0:b0 + batch_size]
            for (v, t), (fr, sc) in zip(ch, loz.predict_multievent_batch([sample_of(v, t) for v, t in ch])):
                for f2, s2 in zip(fr, sc):
                    if 0 <= f2 < v["num_frames"]:
                        pr[v["video"]].append({"label": t, "frame": int(f2), "score": float(s2)})
            if (b0 // batch_size) % 15 == 0:
                print(f"    {b0}/{len(items)}", flush=True)
    return pr


def matched_precision_errs(pr, vids):
    """AP-style one-to-one: per (video,label) sort preds by score desc, greedily match each
    to nearest UNMATCHED GT; err = that distance, or 999 if no GT left (duplicate FP)."""
    errs = []
    for v in vids:
        gts_by, preds_by = {}, {}
        for ev in v["events"]:
            gts_by.setdefault(ev["label"], []).append(ev["frame"])
        for p in pr[v["video"]]:
            preds_by.setdefault(p["label"], []).append(p)
        for lab, preds in preds_by.items():
            gts = list(gts_by.get(lab, [])); used = [False] * len(gts)
            for p in sorted(preds, key=lambda p: -p["score"]):
                best, bj = 10 ** 9, -1
                for j, g in enumerate(gts):
                    if not used[j] and abs(p["frame"] - g) < best:
                        best, bj = abs(p["frame"] - g), j
                if bj >= 0:
                    used[bj] = True; errs.append(best)
                else:
                    errs.append(999)                             # no GT left -> duplicate false-pos
    return np.array(errs)


def recall_errs(pr, vids):
    """Each GT -> nearest same-label prediction (coverage)."""
    errs = []
    for v in vids:
        for ev in v["events"]:
            same = [p["frame"] for p in pr[v["video"]] if p["label"] == ev["label"]]
            errs.append(min((abs(ev["frame"] - p) for p in same), default=999))
    return np.array(errs)


def main():
    dev = "cuda"; en.TOLS = [0, 1, 2]; torch.set_num_threads(8)
    vids = [v for v in json.load(open(os.path.join(LAB, "test.json")))
            if os.path.exists(os.path.join(GRID, v["video"].replace("/", "__") + ".npy"))]
    ngt = sum(len(v["events"]) for v in vids)
    replot = "--replot" in sys.argv and os.path.exists(PRED_CACHE)
    if replot:
        pr = {k: v for k, v in json.load(open(PRED_CACHE)).items()}
        print(f"[precision-dist] loaded cached preds (replot); ngt={ngt}", flush=True)
    else:
        from models.wrapper import QwenWrapper
        print(f"[precision-dist] TEST vids={len(vids)}  model=SFT-LoRA (strongest)", flush=True)
        W = QwenWrapper(device=dev, dtype=torch.bfloat16)
        pr = get_preds(W, vids, dev)
        os.makedirs(os.path.join(ROOT, "plot"), exist_ok=True)
        json.dump(pr, open(PRED_CACHE, "w"))

    a_prec = matched_precision_errs(pr, vids)                     # honest (deduped) precision
    a_rec = recall_errs(pr, vids)                                 # coverage
    # exact AP via the SAME code eval uses -> annotate & confirm reconciliation
    truth = [{"video": v["video"], "num_frames": v["num_frames"],
              "events": [{"label": e["label"], "frame": e["frame"]} for e in v["events"]]} for v in vids]
    predl = [{"video": v["video"], "events": pr[v["video"]]} for v in vids]
    ap = en.maps_quiet(truth, predl)                             # [AP@0, AP@1, AP@2]
    ndet = len(a_prec)

    def cum(a, k):
        return 100 * (a <= k).mean()
    print(f"  ndet={ndet} ngt={ngt} | AP@0/1/2 = {[round(x,2) for x in ap]}", flush=True)
    for k in (0, 1, 2):
        print(f"  @{k}: precision(matched)={cum(a_prec,k):.1f}%  recall(coverage)={cum(a_rec,k):.1f}%  "
              f"AP={ap[k]:.1f}  [check: rec*prec/100={cum(a_rec,k)*cum(a_prec,k)/100:.1f}]", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    fig, (axh, axc) = plt.subplots(1, 2, figsize=(13, 4.8))
    fwd = lambda x: np.sqrt(np.clip(x, 0, None)); inv = lambda x: np.square(x)   # magnify low tol
    ticks = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 30]

    bins = np.arange(0, 32)
    fp = int((a_prec >= 999).sum())
    axh.hist(np.clip(a_prec[a_prec < 999], 0, 31), bins=bins, alpha=0.75, color="#1d7fd1",
             label=f"matched predictions (n={ndet}, dup-FP={fp})")
    axh.set_xscale("function", functions=(fwd, inv)); axh.set_xlim(0, 31)
    axh.set_xticks(ticks); axh.xaxis.set_major_formatter(mticker.ScalarFormatter())
    axh.set_xlabel("prediction distance to its matched GT (frames, sqrt-scaled)"); axh.set_ylabel("count")
    axh.set_title("Precision: distance from each prediction to its one-to-one GT"); axh.legend()

    from common.score import parse_ground_truth, get_predictions, compute_average_precision
    tbl = parse_ground_truth(truth); classes = sorted(tbl.keys())
    def mean_ap(t):                                              # E2E mAP@t (touch/untouch mean)
        return float(np.mean([compute_average_precision(get_predictions(predl, label=lab), tbl[lab],
                                                         tolerance=int(t)) * 100 for lab in classes]))
    xs = np.arange(0, 31)
    ap_curve = [mean_ap(x) for x in xs]
    axc.plot(xs, [cum(a_prec, x) for x in xs], color="#1d7fd1", lw=2.2, label="precision (of preds)")
    axc.plot(xs, [cum(a_rec, x) for x in xs], color="#2ca25f", lw=2.0, ls="--", label="recall (of GT)")
    axc.plot(xs, ap_curve, color="#c0392b", lw=2.6, label="AP / mAP  (E2E eval)")
    axc.set_xscale("function", functions=(fwd, inv)); axc.set_xlim(0, 30); axc.set_ylim(0, 100)
    axc.set_xticks(ticks); axc.xaxis.set_major_formatter(mticker.ScalarFormatter())
    for x in (0, 1, 2, 4, 8):
        axc.axvline(x, color="gray", ls=":", lw=0.6)
    axc.set_xlabel("tolerance (frames, sqrt-scaled)"); axc.set_ylabel("%"); axc.grid(alpha=0.3)
    axc.set_title("Precision, recall & AP vs tolerance (TEST)"); axc.legend(loc="lower right")
    txt = ("per-tolerance (TEST):\n"
           f"      prec  rec   AP\n"
           f"@0    {cum(a_prec,0):>4.0f}  {cum(a_rec,0):>4.0f}  {ap[0]:>4.1f}\n"
           f"@1    {cum(a_prec,1):>4.0f}  {cum(a_rec,1):>4.0f}  {ap[1]:>4.1f}\n"
           f"@2    {cum(a_prec,2):>4.0f}  {cum(a_rec,2):>4.0f}  {ap[2]:>4.1f}\n"
           "AP = area under ranked PR\n"
           "(needs prec & rec BOTH high;\n"
           " @0 AP<<prec: score can't\n"
           " rank exact hits first)")
    axc.text(0.985, 0.46, txt, transform=axc.transAxes, va="center", ha="right", fontsize=8.5,
             family="monospace", bbox=dict(boxstyle="round", fc="#eef5fc", ec="#1d7fd1"))
    fig.suptitle("SFT-LoRA (strongest) — precision / recall / AP vs tolerance  (AP = E2E eval, TEST)",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(ROOT, "plot", "precision_dist_test.png")
    fig.savefig(out, dpi=110); print(f"[plot] {out}", flush=True)


if __name__ == "__main__":
    main()
