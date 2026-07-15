"""P2 infra: run the frozen idx-generation on ALL videos (per type), and write an
idx-PRIOR feature file per video = [F3(768) | touch_prior | untouch_prior] where each
prior channel is a sum of Gaussian bumps at the generated idx frames (weighted by the
generation score). Lets the stage-2 MS-TCN consume the LLM's coarse localization.
Shardable across GPUs. Output: outputs/feat_cache/F3_idxprior/{video}.npy [N,770].
"""
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
F3DIR = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/feat_interleave"
LAB = "/home/chang_noroot/data2/huyanh/Workspace/hand_touch_detection/data/HOI4D-v3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_ckpt", default=os.path.join(ROOT, "outputs/idx/idx_multi/best.pt"))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()
    si, sk = [int(x) for x in args.shard.split("/")]

    from models.wrapper import QwenWrapper
    from models.vjepa_adaptor import VJEPAAdaptor
    from models.idx_localizer import IdxLocalizer
    from data.questions import GENERIC_Q

    ck = torch.load(args.gen_ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]; rank = cfg.get("lora_rank", 16)
    W = QwenWrapper(device="cuda", dtype=torch.bfloat16)
    ad = VJEPAAdaptor(768, W.d_llm, hidden=2048, n_layers=2).to("cuda", torch.bfloat16)
    ad.set_target_rms_from(W.embed_tokens.weight); ad.load_state_dict(ck["adaptor"]); ad.eval()
    if cfg.get("lora"):
        lp = W.add_lora(rank=rank, alpha=2 * rank, n_layers=W.n_layers, target="all")
        for p, s in zip(lp, ck["lora"]):
            p.data.copy_(s.to(p.device, p.dtype))
    loz = IdxLocalizer(W, ad, use_idx=True, use_anchor=True, anchor_stride=5,
                       anchor_max_side=252, use_mrope=False, fps=15, max_frames=320,
                       grad_checkpoint=False)

    vids = set()
    for split in ("train", "val", "test"):
        for v in json.load(open(os.path.join(LAB, f"{split}.json"))):
            k = v["video"].replace("/", "__")
            if os.path.exists(os.path.join(F3DIR, k + ".npy")):
                vids.add(k)
    vids = sorted(vids)[si::sk]
    outd = os.path.join(ROOT, "outputs/feat_cache/F3_idxprior"); os.makedirs(outd, exist_ok=True)
    todo = [k for k in vids if not os.path.exists(os.path.join(outd, k + ".npy"))]
    print(f"[genidx] shard {args.shard} videos={len(todo)}", flush=True)
    types = ("touch", "untouch")

    def bumps(N, frames_scores):
        ch = np.zeros(N, dtype=np.float32)
        xs = np.arange(N)
        for fr, sc in frames_scores:
            ch += sc * np.exp(-((xs - fr) ** 2) / (2 * args.sigma ** 2))
        return ch

    for b0 in range(0, len(todo), args.batch_size):
        chunk = todo[b0:b0 + args.batch_size]
        # one generation per (video,type)
        idx_by = {}
        for t in types:
            samples = []
            for k in chunk:
                f = np.load(os.path.join(F3DIR, k + ".npy")).astype(np.float32); N = f.shape[0]
                samples.append({"feats": torch.from_numpy(f), "question": GENERIC_Q[t],
                                "event_frames": None, "gt": -1, "fps": 15, "type": t,
                                "video_id": k, "num_frames": N, "full_num_frames": N,
                                "anchor_start_sec": 0, "anchor_num_secs": N // 15})
            outs = loz.predict_multievent_batch(samples)
            for k, (fr, sc) in zip(chunk, outs):
                idx_by[(k, t)] = [(int(a), float(b)) for a, b in zip(fr, sc)]
        for k in chunk:
            f = np.load(os.path.join(F3DIR, k + ".npy")).astype(np.float32); N = f.shape[0]
            pri = np.stack([bumps(N, [(a, b) for a, b in idx_by[(k, t)] if 0 <= a < N])
                            for t in types], axis=1)                        # [N,2]
            np.save(os.path.join(outd, k + ".npy"), np.concatenate([f, pri], axis=1).astype(np.float16))
        if (b0 // args.batch_size) % 20 == 0:
            print(f"  {b0}/{len(todo)}", flush=True)
    print(f"[done] shard {args.shard} -> {outd}", flush=True)


if __name__ == "__main__":
    main()
