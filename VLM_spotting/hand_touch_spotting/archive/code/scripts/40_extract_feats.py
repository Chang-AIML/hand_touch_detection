"""Extract & cache per-frame features for the MS-TCN feature ablation:
  F1 = pre-LLM adaptor output (motion tokens as fed to the LLM), [N,4096]
  F2 = post-LLM last-layer hidden at motion positions (LANGUAGE-SPACE), [N,4096]
(F3 = raw V-JEPA [N,768] already exists = feat_interleave.)
Saved as {video__nobs}.npy under outputs/feat_cache/{F1_adaptor,F2_postllm_<q>}/.
Shardable: --shard i/K splits the video list across GPUs.
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
FEAT = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/feat_interleave"
LAB = "/home/chang_noroot/data2/huyanh/Workspace/hand_touch_detection/data/HOI4D-v3"

QUESTIONS = {
    "neutral": "When does the hand make or break contact with the object?",
    "touch": "When does the hand make contact with the object?",
    "untouch": "When does the hand release the object?",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_ckpt", default=os.path.join(ROOT, "outputs/idx/idx_multi/best.pt"))
    ap.add_argument("--qmode", default="neutral", choices=list(QUESTIONS))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--layer", type=int, default=-1, help="LLM layer for F2; -1=last")
    args = ap.parse_args()
    LAYER = None if args.layer == -1 else args.layer
    si, sk = [int(x) for x in args.shard.split("/")]

    from models.wrapper import QwenWrapper
    from models.vjepa_adaptor import VJEPAAdaptor
    from models.idx_localizer import IdxLocalizer

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
            if os.path.exists(os.path.join(FEAT, k + ".npy")):
                vids.add(k)
    vids = sorted(vids)[si::sk]
    lsuf = "" if LAYER is None else f"_L{LAYER}"
    d1 = os.path.join(ROOT, "outputs/feat_cache/F1_adaptor")
    d2 = os.path.join(ROOT, f"outputs/feat_cache/F2_postllm_{args.qmode}{lsuf}")
    os.makedirs(d1, exist_ok=True); os.makedirs(d2, exist_ok=True)
    q = QUESTIONS[args.qmode]
    print(f"[extract] shard {args.shard} qmode={args.qmode} videos={len(vids)}", flush=True)

    todo = [k for k in vids if not (os.path.exists(os.path.join(d1, k + ".npy"))
                                    and os.path.exists(os.path.join(d2, k + ".npy")))]
    for b0 in range(0, len(todo), args.batch_size):
        chunk = todo[b0:b0 + args.batch_size]
        batch = []
        for k in chunk:
            f = np.load(os.path.join(FEAT, k + ".npy")).astype(np.float32)
            N = f.shape[0]
            batch.append({"feats": torch.from_numpy(f), "question": q, "fps": 15,
                          "video_id": k, "num_frames": N, "full_num_frames": N,
                          "anchor_start_sec": 0, "anchor_num_secs": N // 15})
        outs = loz.extract_frame_features(batch, layer=LAYER)
        for k, (F1, F2) in zip(chunk, outs):
            np.save(os.path.join(d1, k + ".npy"), F1.numpy().astype(np.float16))
            np.save(os.path.join(d2, k + ".npy"), F2.numpy().astype(np.float16))
        if (b0 // args.batch_size) % 20 == 0:
            print(f"  {b0}/{len(todo)}", flush=True)
    print(f"[done] shard {args.shard} -> {d1} , {d2}", flush=True)


if __name__ == "__main__":
    main()
