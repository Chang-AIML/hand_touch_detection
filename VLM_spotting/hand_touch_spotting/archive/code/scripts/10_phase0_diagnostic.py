"""Phase 0 feasibility diagnostic [GATE 1] — exp_plan §4.

A1: after the frozen LLM, are the V-JEPA hidden states still temporally
discriminative at the contact frame? Two confounds must be avoided:
  * an *untrained* query can't localise -> query-sim looks flat (uninformative);
  * a *random* adaptor feeds the LLM OOD noise -> post-LLM hiddens collapse.

So the PRIMARY, training-free probe is the SELF-PROTOTYPE distinctiveness curve
  d(t) = cos( f(gt), f(t) )   aligned to GT, averaged over events,
measured on (i) raw V-JEPA feats [pre-LLM baseline] and (ii) the post-LLM hidden
at each candidate layer. A distinctive contact frame => d(t) peaks at the centre
(offset 0 by construction) and falls off quickly => small FWHM. If the frozen LLM
smears everything, d(t) stays ~1 everywhere => large FWHM => A1 fails.

The V-JEPA tokens are injected through a STRUCTURE-PRESERVING projection (fixed
semi-orthogonal 768->d_llm, cosine-preserving, dense, scaled to Qwen's embedding
RMS) so the LLM sees a faithful embedding of V-JEPA rather than random noise.

Secondary: query-sim s(t)=cos(h_query, h_vjepa(t)) with the untrained LOC / last
question token (expected weak; reported for reference).

GATE 1 (§4.4, adapted): GO if some layer's self-prototype FWHM <= 4 AND the raw
pre-LLM FWHM is also small (signal exists and survives); WEAK-GO if <= 8; NO-GO if
all layers smear (FWHM > 8) while raw is sharp (=> LLM destroys the signal).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/home/chang_noroot/data2/hf_cache")

from localize.sim_readout import align_to_gt, compute_fwhm, cosine_curve  # noqa: E402

FRAMES_DIR = os.environ.get(
    "TOUCH_FRAMES_DIR",
    "/home/chang_noroot/data2/huyanh/Workspace/dataset/hoi4d/frames")


def load_1fps_frames(video_id, num_frames, fps):
    import glob
    from PIL import Image
    paths = sorted(glob.glob(os.path.join(FRAMES_DIR, video_id, "*.jpg")))
    secs = list(range(0, num_frames, fps))
    return [np.asarray(Image.open(paths[min(s, len(paths) - 1)]).convert("RGB")) for s in secs]


def make_orth_proj(d_in, d_out, target_rms, seed=0, device="cuda", dtype=torch.bfloat16):
    """Fixed semi-orthogonal (d_out,d_in) so columns are orthonormal -> cosine of
    inputs is preserved (up to global scale). Output later rescaled per-token to
    target_rms so injected tokens match Qwen's input-embedding scale."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(d_out, d_in, generator=g)
    q, _ = torch.linalg.qr(a)                       # (d_out, d_in) semi-orthogonal cols
    return q.to(device, dtype), float(target_rms)


def project(feats_bf16, Q, target_rms):
    """feats: (N,768) -> (N,d_llm) cosine-preserving, per-token RMS = target_rms."""
    y = feats_bf16 @ Q.T                            # (N, d_llm)
    rms = y.float().pow(2).mean(-1, keepdim=True).clamp_min(1e-12).sqrt()
    return (y / rms.to(y.dtype) * target_rms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--feat-dir", default=os.path.join(ROOT, "outputs", "vjepa_feats"))
    ap.add_argument("--layers", type=int, nargs="+", default=[6, 12, 18, 24, 30, 36])
    ap.add_argument("--max-events", type=int, default=200)
    ap.add_argument("--half", type=int, default=15)
    ap.add_argument("--anchor", action="store_true")
    ap.add_argument("--types", nargs="+", default=["touch", "untouch"])
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    from models.wrapper import QwenWrapper
    from models.loc_tokens import LocTokens

    dev = "cuda"
    W = QwenWrapper(device=dev, dtype=torch.bfloat16)
    target_rms = float(W.embed_tokens.weight.float().pow(2).mean(-1).sqrt().mean())
    Q, target_rms = make_orth_proj(768, W.d_llm, target_rms, seed=args.seed, device=dev)
    loc = LocTokens(W.d_llm, k=1).to(dev, torch.bfloat16)
    loc.init_from_embeddings(W.embed_tokens.weight)

    samples = json.load(open(os.path.join(ROOT, "data", "annotations", f"{args.split}_samples.json")))
    samples = [s for s in samples if s["type"] in args.types
               and os.path.exists(os.path.join(args.feat_dir, s["video_id"] + ".npy"))]

    acc = defaultdict(list)     # (metric, type) -> [aligned curve]
    n_used = 0
    for s in samples:
        if n_used >= args.max_events:
            break
        vid, q, gt = s["video_id"], s["question"], s["frame"]
        nf, fps, typ = s["num_frames"], s["fps"], s["type"]
        feats = np.load(os.path.join(args.feat_dir, vid + ".npy")).astype(np.float32)
        vj = torch.from_numpy(feats).to(dev, torch.bfloat16)

        # ---- pre-LLM self-prototype on raw V-JEPA ----
        raw_self = cosine_curve(vj[gt], vj).cpu().numpy()
        for tt in (typ, "all"):
            acc[("selfproto_raw", tt)].append(align_to_gt(raw_self, gt, args.half))

        with torch.no_grad():
            vj_e = project(vj, Q, target_rms)
            ag = W.vit_anchor_groups(load_1fps_frames(vid, nf, fps)) if args.anchor else None
            r = W.build_and_run(vj_e, q, loc.loc_embeds(torch.bfloat16, dev), nf, fps,
                                anchor_groups=ag)
        hs = r["hidden_states"]
        vpos = r["vjepa_pos"].to(dev)
        loc_p = int(r["loc_pos"][0]); q_last = loc_p - 1

        for L in args.layers:
            h = hs[L][0]
            hvj = h[vpos]                                   # (N, d)
            # PRIMARY: post-LLM self-prototype distinctiveness
            selfp = cosine_curve(hvj[gt], hvj).cpu().numpy()
            for tt in (typ, "all"):
                acc[(f"selfproto_L{L}", tt)].append(align_to_gt(selfp, gt, args.half))
            # SECONDARY: untrained query-sim
            for qkey, qpos in [("qsim_qtok", q_last), ("qsim_loc", loc_p)]:
                s_t = cosine_curve(h[qpos], hvj).cpu().numpy()
                for tt in (typ, "all"):
                    acc[(f"{qkey}_L{L}", tt)].append(align_to_gt(s_t, gt, args.half))
        n_used += 1
        if n_used % 25 == 0:
            print(f"  processed {n_used} events", flush=True)

    # aggregate
    rows, curves_out = [], {}
    for (key, typ), lst in sorted(acc.items()):
        arr = np.vstack(lst)
        mean = np.nanmean(arr, axis=0)
        curves_out[f"{key}|{typ}"] = mean.tolist()
        fwhm, off, pk = compute_fwhm(mean)
        rows.append({"metric": key, "type": typ, "n_events": arr.shape[0],
                     "fwhm": round(fwhm, 2), "offset": round(off, 2), "peak": round(pk, 4)})

    import csv
    tag = "anchor" if args.anchor else "noanchor"
    csv_path = os.path.join(args.out, f"phase0_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=["metric", "type", "n_events", "fwhm", "offset", "peak"])
        wtr.writeheader(); wtr.writerows(rows)
    json.dump(curves_out, open(csv_path.replace(".csv", "_curves.json"), "w"))
    print(f"\n[phase0-{tag}] {n_used} events | wrote {csv_path}")

    def show(prefix, title):
        print(f"\n=== {title} (type=all) ===")
        for r in rows:
            if r["metric"].startswith(prefix) and r["type"] == "all":
                print(f"  {r['metric']:16s} FWHM={r['fwhm']:6} offset={r['offset']:6} peak={r['peak']}")
    show("selfproto", "SELF-PROTOTYPE distinctiveness FWHM  (PRIMARY: raw=pre-LLM, L*=post-LLM)")
    show("qsim", "query-sim FWHM (secondary, untrained)")

    # GATE verdict on post-LLM self-prototype
    raw_row = next((r for r in rows if r["metric"] == "selfproto_raw" and r["type"] == "all"), None)
    post = [r for r in rows if r["metric"].startswith("selfproto_L") and r["type"] == "all"
            and not np.isnan(r["fwhm"])]
    if post:
        best = min(post, key=lambda r: r["fwhm"])
        v = best["fwhm"]
        verdict = "GO" if v <= 4 else ("WEAK-GO" if v <= 8 else "NO-GO")
        print(f"\n=== GATE 1: {verdict} ===")
        print(f"  best post-LLM self-prototype: {best['metric']} FWHM={v}")
        if raw_row:
            print(f"  raw (pre-LLM) self-prototype FWHM={raw_row['fwhm']} "
                  f"-> signal {'EXISTS' if raw_row['fwhm'] <= 12 else 'weak'} pre-LLM; "
                  f"LLM {'preserves' if v <= raw_row['fwhm'] + 3 else 'broadens'} it")


if __name__ == "__main__":
    main()
