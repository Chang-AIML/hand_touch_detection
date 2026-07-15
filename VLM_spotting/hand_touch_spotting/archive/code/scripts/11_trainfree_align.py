"""Training-free alignment smoke test (per user's redirection).

Frozen VLM, NO decode head, NO training. Question: does the frozen VLM's LATENT
for "when does contact happen" naturally retrieve the V-JEPA token at the contact
frame? We inject V-JEPA tokens through a FIXED structure-preserving projection
(cosine-preserving, no training), run frozen Qwen, and at each layer L compare a
query-latent h[probe] against the per-frame V-JEPA hiddens h[vjepa_t]:

    s(t) = cos(h[probe], h[vjepa_t]) ;  pred = argmax_t s(t)

Report per-event MAE (frames) and hit@{0,1,2,4} for each (probe, layer). Also the
ORACLE self-prototype cos(h[vjepa_gt], h[vjepa_t]) to confirm V-JEPA token
discriminability is preserved through the network (区分度是否保持).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/home/chang_noroot/data2/hf_cache")

TOLS = [0, 1, 2, 4]
LAYERS_DEFAULT = [6, 12, 18, 24, 30, 36]
# probe = an appended phrase; we read the hidden at its LAST token (and its mean)
PROMPT_SUFFIX = " Answer: the exact frame of that moment is"


def make_orth_proj(d_in, d_out, target_rms, seed, device, dtype):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d_out, d_in, generator=g))
    return q.to(device, dtype), float(target_rms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--feat-dir", default="/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/feat_interleave")
    ap.add_argument("--layers", type=int, nargs="+", default=LAYERS_DEFAULT)
    ap.add_argument("--max-events", type=int, default=150)
    ap.add_argument("--types", nargs="+", default=["touch", "untouch"])
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--anchor", action="store_true", help="inject frozen Qwen-ViT 1fps anchors (VLM sees video)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from models.wrapper import QwenWrapper
    from data.token_layout import assemble_embeds
    import glob
    from PIL import Image
    FRAMES = os.environ.get("TOUCH_FRAMES_DIR",
                            "/home/chang_noroot/data2/huyanh/Workspace/dataset/hoi4d/frames")

    def frames_1fps(vid, N, fps):
        paths = sorted(glob.glob(os.path.join(FRAMES, vid, "*.jpg")))
        return [np.asarray(Image.open(paths[min(s, len(paths) - 1)]).convert("RGB"))
                for s in range(0, N, fps)]

    dev = "cuda"
    W = QwenWrapper(device=dev, dtype=torch.bfloat16)
    trms = float(W.embed_tokens.weight.float().pow(2).mean(-1).sqrt().mean())
    Q, trms = make_orth_proj(768, W.d_llm, trms, 0, dev, torch.bfloat16)

    samples = json.load(open(os.path.join(ROOT, "data", "annotations", f"{args.split}_samples.json")))
    samples = [s for s in samples if s["type"] in args.types
               and os.path.exists(os.path.join(args.feat_dir, s["video_id"] + ".npy"))][:args.max_events]

    # accumulators: (probe, layer) -> list of (abs_err)
    errs = defaultdict(list)
    oracle_err = defaultdict(list)   # layer -> 2nd-best offset (discriminability)

    tok = W.tokenizer
    for ei, s in enumerate(samples):
        vid, q, gt = s["video_id"], s["question"], s["frame"]
        feats = np.load(os.path.join(args.feat_dir, vid + ".npy")).astype(np.float32)
        N = feats.shape[0]
        vj = torch.from_numpy(feats).to(dev, torch.bfloat16)
        with torch.no_grad():
            vj_e = vj @ Q.T
            vj_e = vj_e / vj_e.float().pow(2).mean(-1, keepdim=True).clamp_min(1e-12).sqrt().to(vj_e.dtype) * trms
            prompt = q + PROMPT_SUFFIX
            pids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
            p_e = W.embed_tokens(pids)[0]                       # (P, d)
            loc_dummy = p_e[-1:]                                # reuse last prompt token as trailing probe slot
            if args.anchor:
                ag = W.vit_anchor_groups(frames_1fps(vid, N, s["fps"]))
                lay = assemble_embeds(ag, vj_e, p_e[:-1], loc_dummy, N, s["fps"])
                seq = lay["embeds"]
                vpos = lay["vjepa_pos"].to(dev)
                probe_last = int(lay["loc_pos"][0])
            else:
                seq = torch.cat([vj_e, p_e], 0)                # [vjepa | prompt]
                vpos = torch.arange(N, device=dev)
                probe_last = seq.shape[0] - 1
            hs = W.run(seq)                                     # all layers
        for L in args.layers:
            h = hs[L][0]
            hvj = h[vpos].float()                              # (N,d)
            hvjn = F.normalize(hvj, dim=-1)
            q_last = F.normalize(h[probe_last].float(), dim=-1)
            s_t = hvjn @ q_last
            errs[("last", L)].append(abs(int(s_t.argmax()) - gt))
            # raw-VJEPA cross-space: project query 4096->768 via Q, cos with raw feats
            q768 = F.normalize((h[probe_last].float() @ Q.float()), dim=-1)
            rawn = F.normalize(vj.float(), dim=-1)
            errs[("last_raw768", L)].append(abs(int((rawn @ q768).argmax()) - gt))
            # oracle self-proto: is gt distinctive? 2nd-best neighbour offset
            so = hvjn @ hvjn[gt]; so[gt] = -9
            oracle_err[L].append(abs(int(so.argmax()) - gt))
        if (ei + 1) % 25 == 0:
            print(f"  {ei+1}/{len(samples)}", flush=True)

    # report
    def rates(v):
        v = np.array(v)
        return {"MAE": float(v.mean()), **{f"hit@{t}": float((v <= t).mean() * 100) for t in TOLS}}
    print(f"\n=== TRAINING-FREE alignment ({len(samples)} events, split {args.split}) ===")
    print(f"{'probe':11s} {'L':>3s} {'MAE':>7s} {'hit@0':>7s} {'hit@1':>7s} {'hit@2':>7s} {'hit@4':>7s}")
    rows = []
    for (name, L), v in sorted(errs.items(), key=lambda x: (x[0][0], x[0][1])):
        r = rates(v)
        rows.append((name, L, r))
        print(f"{name:11s} {L:3d} {r['MAE']:7.2f} {r['hit@0']:7.1f} {r['hit@1']:7.1f} "
              f"{r['hit@2']:7.1f} {r['hit@4']:7.1f}")
    print("\n=== ORACLE self-proto (VJEPA discriminability: 2nd-best neighbour |offset|) ===")
    for L in args.layers:
        v = np.array(oracle_err[L])
        print(f"  L{L:2d}: 2nd-best within 1 frame: {(v <= 1).mean()*100:.1f}% | median offset {np.median(v):.0f}")
    best = min(rows, key=lambda r: r[2]["MAE"])
    print(f"\nbest training-free: probe={best[0]} L={best[1]} MAE={best[2]['MAE']:.2f} "
          f"hit@2={best[2]['hit@2']:.1f}%  (random hit@2 ~ 1.7%)")
    json.dump([{"probe": n, "L": L, **r} for n, L, r in rows],
              open(os.path.join(args.out, "trainfree_align.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
