"""Per-event localization-error figure for the NEW compression pipeline (analog of the old
refine_error_dist_test.png, but Stage-1 idx-gen only). Compares the frozen-compression
connector (M1, 64.3 test) vs the strongest SFT-LoRA (72.9 test) on TEST: histogram of
|pred-GT| per GT event + cumulative-recall CDF vs tolerance. Shows SFT pulling error mass
toward 0 (esp. the @0/@1 exact bins). Reuses the scripts/68 test-eval prediction path."""
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

MODELS = [("frozen-compress (M1)", "outputs/idx_compress/comp_notext/best.pt", "#9aa0a6"),
          ("+ SFT-LoRA (strongest)", "outputs/idx_compress/sft_lora_v2/best.pt", "#1d7fd1")]


def preds_for(ckpt, W, vids, dev, batch_size=8):
    from models.vjepa_adaptor import VJEPAAdaptor
    from models.idx_localizer import IdxLocalizer
    from models.frame_compress import FrameCompress
    ck = torch.load(ckpt, weights_only=False)
    ad = VJEPAAdaptor(768, W.d_llm, hidden=2048, n_layers=2).to(dev, torch.bfloat16)
    fc = FrameCompress(768, 8, W.d_llm, n_heads=4, stain=False, use_text=False).to(dev, torch.float32)
    fc.set_target_rms_from(W.embed_tokens.weight); fc.load_state_dict(ck["fc"]); fc.eval()
    lp = None
    if "lora" in ck:
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
    if lp is not None:                                            # remove LoRA so next model is clean
        W.remove_lora() if hasattr(W, "remove_lora") else None
    return pr


def errs_of(pr, vids):
    e = []
    for v in vids:
        preds = pr[v["video"]]
        for ev in v["events"]:
            same = [p["frame"] for p in preds if p["label"] == ev["label"]]
            e.append(min((abs(ev["frame"] - p) for p in same), default=999))
    return np.array(e)


def main():
    dev = "cuda"; en.TOLS = [0, 1, 2]; torch.set_num_threads(8)
    from models.wrapper import QwenWrapper
    vids = [v for v in json.load(open(os.path.join(LAB, "test.json")))
            if os.path.exists(os.path.join(GRID, v["video"].replace("/", "__") + ".npy"))]
    print(f"[error-dist] TEST vids={len(vids)}", flush=True)

    results = []
    for name, ckpt, color in MODELS:
        print(f"  running {name} ...", flush=True)
        W = QwenWrapper(device=dev, dtype=torch.bfloat16)         # fresh W per model (clean LoRA state)
        pr = preds_for(os.path.join(ROOT, ckpt), W, vids, dev)
        a = errs_of(pr, vids)
        results.append((name, color, a))
        del W; torch.cuda.empty_cache()

    def stats(a):
        h = a[a < 999]
        return {"n": len(a), "miss": 100 * (a >= 999).mean(), "median": np.median(h) if len(h) else 0,
                **{f"w{k}": 100 * (a <= k).mean() for k in (0, 1, 2, 4, 8)}}

    for name, _, a in results:
        print(f"  {name}: {  {k: round(v, 1) for k, v in stats(a).items()} }", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (axh, axc) = plt.subplots(1, 2, figsize=(13, 4.6))
    bins = np.arange(0, 32)
    for name, color, a in results:
        S = stats(a)
        axh.hist(np.clip(a[a < 999], 0, 31), bins=bins, alpha=0.55, color=color,
                 label=f"{name}  (median {S['median']:.0f}f, @0 {S['w0']:.0f}%)")
    axh.set_xlabel("localization error |pred - GT| (frames)"); axh.set_ylabel("count")
    axh.set_title("Per-event error: SFT-LoRA pulls mass to the @0/@1 bins"); axh.legend()

    xs = np.arange(0, 31)
    for name, color, a in results:
        axc.plot(xs, [100 * (a <= x).mean() for x in xs], color=color, lw=2.2, label=name)
    for x in (0, 1, 2, 4, 8):
        axc.axvline(x, color="gray", ls=":", lw=0.6)
    axc.set_xlabel("tolerance (frames)"); axc.set_ylabel("% GT events within tolerance")
    axc.set_title("Cumulative recall vs tolerance (TEST)"); axc.grid(alpha=0.3); axc.legend(loc="lower right")
    S0, S1 = stats(results[0][2]), stats(results[1][2])
    txt = (f"within 0f: {S0['w0']:.0f}% -> {S1['w0']:.0f}%\n"
           f"within 1f: {S0['w1']:.0f}% -> {S1['w1']:.0f}%\n"
           f"within 2f: {S0['w2']:.0f}% -> {S1['w2']:.0f}%")
    axc.text(0.03, 0.97, txt, transform=axc.transAxes, va="top", fontsize=9,
             family="monospace", bbox=dict(boxstyle="round", fc="#eef5fc", ec="#1d7fd1"))
    fig.suptitle("Query-compression Stage-1 idx-gen — per-event localization error "
                 "(TEST, M1 64.3 -> SFT-LoRA 72.9 mAP@2)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(ROOT, "plot", "compress_error_dist_test.png")
    os.makedirs(os.path.join(ROOT, "plot"), exist_ok=True)
    fig.savefig(out, dpi=110); print(f"[plot] {out}", flush=True)


if __name__ == "__main__":
    main()
