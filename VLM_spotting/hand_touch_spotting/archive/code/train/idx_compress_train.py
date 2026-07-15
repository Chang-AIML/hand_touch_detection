"""Stage-1 trainer to test LANGUAGE INJECTION at the V-JEPA->LLM connection.
FROZEN LLM (no LoRA, Phase-1 alignment). Trainable = the front-end only:
  --mode compress : FrameCompress (3-step cross-attn, question-conditioned) over grid_even
  --mode mean     : VJEPAAdaptor over mean-pooled feat_interleave  (baseline)
Eval = full-video multi-event idx generation -> mAP@0/1/2 (recall via ndet/ngt).
OOM-safe: small batch, grad-checkpoint LLM, synchronous data (no worker procs), thread cap.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/home/chang_noroot/data2/hf_cache")
_COMMON = "/home/chang_noroot/data2/huyanh/Workspace/hand_touch_detection"
sys.path.insert(0, _COMMON); sys.path.insert(0, os.path.join(_COMMON, "methods/spot_head"))
FEAT = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/feat_interleave"
LAB = os.path.join(_COMMON, "data", "HOI4D-v3")
import eval_nms as en                                             # noqa: E402
from data.questions import GENERIC_Q                              # noqa: E402


def build_truth(vids):
    return [{"video": v["video"], "num_frames": v["num_frames"],
             "events": [{"label": e["label"], "frame": e["frame"]} for e in v["events"]]} for v in vids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="compress", choices=["compress", "mean"])
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--stain", type=int, default=1); ap.add_argument("--n_q", type=int, default=8)
    ap.add_argument("--use_text", type=int, default=1); ap.add_argument("--gate_lang", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--train_limit", type=int, default=0)
    ap.add_argument("--eval_videos", type=int, default=140); ap.add_argument("--cpu_threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora", type=int, default=0); ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--lr_lora", type=float, default=1e-4)
    ap.add_argument("--init_fc", default="")     # load Phase-1 compressor (LLaVA staged Phase-2)
    a = ap.parse_args()
    torch.manual_seed(a.seed); torch.set_num_threads(a.cpu_threads); dev = "cuda"
    en.TOLS = [0, 1, 2]

    from models.wrapper import QwenWrapper
    from models.vjepa_adaptor import VJEPAAdaptor
    from models.idx_localizer import IdxLocalizer
    from models.frame_compress import FrameCompress
    from data.idx_dataset import IdxMultiEventDataset
    from data.idx_grid_dataset import IdxMultiGridDataset

    W = QwenWrapper(device=dev, dtype=torch.bfloat16)
    ad = VJEPAAdaptor(768, W.d_llm, hidden=2048, n_layers=2).to(dev, torch.bfloat16)
    ad.set_target_rms_from(W.embed_tokens.weight)
    fc = None
    if a.mode == "compress":
        fc = FrameCompress(768, a.n_q, W.d_llm, n_heads=4, stain=bool(a.stain),
                           use_text=bool(a.use_text), gate_lang=bool(a.gate_lang)).to(dev, torch.float32)
        fc.set_target_rms_from(W.embed_tokens.weight)
        if a.init_fc:                                            # LLaVA staged: start Phase-2 from Phase-1 compressor
            sd = torch.load(a.init_fc, weights_only=False)["fc"]
            miss, unexp = fc.load_state_dict(sd, strict=False)
            print(f"[init] loaded Phase-1 fc from {a.init_fc} (missing {len(miss)} unexpected {len(unexp)})", flush=True)
        trainable = list(fc.parameters()); train_ds = IdxMultiGridDataset("train")
    else:
        for p in ad.parameters():
            p.requires_grad_(True)
        trainable = list(ad.parameters()); train_ds = IdxMultiEventDataset("train", FEAT)
    front_params = trainable
    lora_params = []
    if a.lora:                                                    # Phase-2: unfreeze LLM via LoRA
        lora_params = W.add_lora(rank=a.lora_rank, alpha=2 * a.lora_rank, n_layers=W.n_layers, target="all")
        trainable = front_params + lora_params
    loz = IdxLocalizer(W, ad, use_idx=True, use_anchor=True, anchor_stride=5, anchor_max_side=252,
                       use_mrope=False, fps=15, max_frames=320, grad_checkpoint=True, compress=fc)
    tp = sum(p.numel() for p in trainable) / 1e6
    print(f"[cfg] mode={a.mode} stain={a.stain} n_q={a.n_q} use_text={a.use_text} | "
          f"trainable={tp:.2f}M | train {len(train_ds)} | bs={a.batch_size}", flush=True)

    # eval videos (from HOI4D-v3, that have grids/feats)
    grid_dir = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/grid_even"
    need = grid_dir if a.mode == "compress" else FEAT
    val_vids = [v for v in json.load(open(os.path.join(LAB, "val.json")))
                if os.path.exists(os.path.join(need, v["video"].replace("/", "__") + ".npy"))][:a.eval_videos]
    truth = build_truth(val_vids)
    types = ("touch", "untouch")
    val_items = [(v, t) for v in val_vids for t in types if any(e["label"] == t for e in v["events"])]

    def val_sample(v, t):
        k = v["video"].replace("/", "__")
        if a.mode == "compress":
            g = np.load(os.path.join(grid_dir, k + ".npy")); N = min(v["num_frames"], 2 * g.shape[0])
            return {"grid": torch.from_numpy(g), "question": GENERIC_Q[t],
                    "type": t, "video_id": k, "fps": 15, "num_frames": N, "full_num_frames": N,
                    "anchor_start_sec": 0, "anchor_num_secs": N // 15, "event_frames": None, "gt": -1}
        f = np.load(os.path.join(FEAT, k + ".npy")).astype(np.float32); N = f.shape[0]
        return {"feats": torch.from_numpy(f), "question": GENERIC_Q[t],
                "type": t, "video_id": k, "fps": 15, "num_frames": N, "full_num_frames": N,
                "anchor_start_sec": 0, "anchor_num_secs": N // 15, "event_frames": None, "gt": -1}

    @torch.no_grad()
    def evaluate():
        loz.W.model.eval()
        if fc is not None:
            fc.eval()
        pr = {v["video"]: [] for v in val_vids}; ndet = 0
        for b0 in range(0, len(val_items), a.batch_size):
            chunk = val_items[b0:b0 + a.batch_size]
            samples = [val_sample(v, t) for v, t in chunk]
            for (v, t), (fr, sc) in zip(chunk, loz.predict_multievent_batch(samples)):
                Nf = v["num_frames"]
                for f2, s2 in zip(fr, sc):
                    if 0 <= f2 < Nf:
                        pr[v["video"]].append({"label": t, "frame": int(f2), "score": float(s2)}); ndet += 1
        pl = [{"video": v["video"], "events": pr[v["video"]]} for v in val_vids]
        m = en.maps_quiet(truth, pl)
        ngt = sum(len(v["events"]) for v in val_vids)
        return m, ndet, ngt

    groups = [{"params": front_params, "lr": a.lr}]
    if lora_params:
        groups.append({"params": lora_params, "lr": a.lr_lora})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    idxs = list(range(len(train_ds)))
    if a.train_limit:
        random.Random(a.seed).shuffle(idxs); idxs = idxs[:a.train_limit]
    spe = len(idxs) // a.batch_size
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, spe * a.epochs))
    out_dir = os.path.join(ROOT, "outputs", "idx_compress", a.run_name); os.makedirs(out_dir, exist_ok=True)
    mp = os.path.join(out_dir, "metrics.csv"); open(mp, "w").write("epoch,loss,mAP0,mAP1,mAP2,ndet,ngt\n")

    m0, nd0, ng0 = evaluate()
    print(f"[ep-1 init] mAP@0/1/2={[round(x,2) for x in m0]} ndet={nd0} ngt={ng0}", flush=True)
    best = -1.0
    for ep in range(a.epochs):
        if fc is not None:
            fc.train()
        rng = random.Random(1000 + ep); order = idxs[:]; rng.shuffle(order)
        rl, t0, nb = 0.0, time.time(), 0
        for b0 in range(0, spe * a.batch_size, a.batch_size):
            batch = [train_ds[i] for i in order[b0:b0 + a.batch_size]]
            loss = loz.loss_batch(batch)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); sched.step()
            rl += float(loss.detach()); nb += 1
            if nb % 50 == 0:
                print(f"  ep{ep} step{nb}/{spe} loss {rl/nb:.4f} | {nb*a.batch_size/(time.time()-t0):.1f} samp/s", flush=True)
        m, nd, ng = evaluate()
        print(f"[ep{ep}] loss {rl/max(nb,1):.4f} | mAP@0/1/2={[round(x,2) for x in m]} ndet={nd} ngt={ng} "
              f"| {len(order)/(time.time()-t0):.1f} samp/s", flush=True)
        open(mp, "a").write(f"{ep},{rl/max(nb,1):.4f},{m[0]:.2f},{m[1]:.2f},{m[2]:.2f},{nd},{ng}\n")
        if m[2] > best:
            best = m[2]
            if fc is not None:
                ck = {"fc": fc.state_dict() if fc is not None else None, "m": m}
                if lora_params:
                    ck["lora"] = [p.detach().cpu() for p in lora_params]
                torch.save(ck, os.path.join(out_dir, "best.pt"))
    print(f"[done] {a.run_name} best mAP@2 {best:.2f}", flush=True)


if __name__ == "__main__":
    main()
