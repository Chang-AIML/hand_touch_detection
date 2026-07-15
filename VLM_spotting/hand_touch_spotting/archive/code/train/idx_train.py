"""Train the idx-decode localizer: the LLM outputs the frame index as digit tokens.

Trainable: V-JEPA adaptor (always) + optional LoRA on ALL 36 LLM layers.
Eval (single-event windows): greedy-generate the index, score hit@{0,1,2,4} + MAE
+ parse-fail rate. This is the copy/count sanity; multi-event mAP is a separate pass.

Single-GPU by design (pin with CUDA_VISIBLE_DEVICES) so the {idx,no-idx}x{frozen,lora}
matrix can run concurrently across the two GPUs.
"""
from __future__ import annotations

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

from data.idx_dataset import IdxSingleEventDataset, IdxMultiEventDataset   # noqa: E402


def _subsample(ds, limit, seed=0):
    if not limit or limit >= len(ds):
        return list(range(len(ds)))
    rng = random.Random(seed)
    return rng.sample(range(len(ds)), limit)


@torch.no_grad()
def evaluate(loz, ds, idxs, batch_size, tolerances=(0, 1, 2, 4), log_n=6):
    loz.W.model.eval(); loz.ad.eval()
    errs, fails, examples = [], 0, []
    clean = 0
    for b0 in range(0, len(idxs), batch_size):
        batch = [ds[i] for i in idxs[b0:b0 + batch_size]]
        preds, texts = loz.predict_batch(batch)
        for s, p, t in zip(batch, preds, texts):
            clean += (s["n_same_in_window"] == 1)
            if p < 0:
                fails += 1
                errs.append(999)
            else:
                errs.append(abs(p - s["gt"]))
            if len(examples) < log_n:
                examples.append((s["gt"], p, s["n_same_in_window"]))
    errs = np.array(errs)
    hit = {t: float((errs <= t).mean()) * 100 for t in tolerances}
    mae = float(np.median(errs[errs < 999])) if (errs < 999).any() else 999.0
    mean_ae = float(errs[errs < 999].mean()) if (errs < 999).any() else 999.0
    return {"hit": hit, "medAE": mae, "MAE": mean_ae,
            "fail%": 100 * fails / len(idxs), "clean%": 100 * clean / len(idxs),
            "examples": examples}


def setup_dist():
    """(distributed, rank, world, local, device). torchrun sets WORLD_SIZE>1."""
    import torch.distributed as dist
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        return True, rank, world, local, f"cuda:{local}"
    return False, 0, 1, 0, "cuda"


def run(use_idx=True, use_anchor=True, anchor_stride=1, anchor_max_side=168, use_mrope=False,
        lora=True, lora_rank=16, lr_adaptor=1e-3, lr_lora=1e-4, epochs=6, batch_size=12,
        crop_secs=6, train_limit=4000, val_limit=600, types=("touch", "untouch"),
        run_name="idx_run", seed=0, multi_event=False, eval_videos=200):
    import torch.distributed as dist
    from models.wrapper import QwenWrapper
    from models.vjepa_adaptor import VJEPAAdaptor
    from models.idx_localizer import IdxLocalizer

    distributed, rank, world, local, dev = setup_dist()
    is_main = rank == 0
    torch.manual_seed(seed)                       # identical init across ranks
    FEAT = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/feat_interleave"
    fps = 15

    W = QwenWrapper(device=dev, dtype=torch.bfloat16)
    ad = VJEPAAdaptor(768, W.d_llm, hidden=2048, n_layers=2).to(dev, torch.bfloat16)
    ad.set_target_rms_from(W.embed_tokens.weight)
    for p in ad.parameters():
        p.requires_grad_(True)
    lora_params = []
    if lora:
        lora_params = W.add_lora(rank=lora_rank, alpha=2 * lora_rank,
                                 n_layers=W.n_layers, target="all")
    loz = IdxLocalizer(W, ad, use_idx=use_idx, use_anchor=use_anchor,
                       anchor_stride=anchor_stride, anchor_max_side=anchor_max_side,
                       use_mrope=use_mrope, fps=fps, max_frames=320, grad_checkpoint=True)
    trainable = list(ad.parameters()) + lora_params

    tp = sum(p.numel() for p in trainable)
    if is_main:
        window = "full-video(300f)" if multi_event else f"crop={crop_secs}s({crop_secs*fps}f)"
        print(f"[cfg] run={run_name} mode={'multi-event' if multi_event else 'single-event'} "
              f"{window} use_idx={use_idx} anchor={use_anchor}@{anchor_max_side}px "
              f"stride={anchor_stride} mrope={use_mrope} lora={lora} rank={lora_rank} "
              f"world={world} trainable={tp/1e6:.2f}M bs={batch_size}(eff {batch_size*world})", flush=True)

    if multi_event:
        train_ds = IdxMultiEventDataset("train", FEAT, types=types, question_phase="train")
        val_ds = None
    else:
        train_ds = IdxSingleEventDataset("train", FEAT, types=types, crop_secs=crop_secs,
                                         question_phase="train", deterministic=False)
        val_ds = IdxSingleEventDataset("val", FEAT, types=types, crop_secs=crop_secs,
                                       question_phase="test", deterministic=True)
    tr_idx = _subsample(train_ds, train_limit, seed)     # same list across ranks
    va_idx = _subsample(val_ds, val_limit, 123) if val_ds is not None else []
    if is_main:
        vn = f"multievent(full-video, {eval_videos} vids)" if multi_event else len(va_idx)
        print(f"[data] train {len(tr_idx)}/{len(train_ds)} | val {vn}", flush=True)

    groups = [{"params": list(ad.parameters()), "lr": lr_adaptor}]
    if lora_params:
        groups.append({"params": lora_params, "lr": lr_lora})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    steps_per_epoch = (len(tr_idx) // world) // batch_size      # per-rank shard
    total = steps_per_epoch * epochs
    warm = int(0.05 * total)

    def lr_scale(s):
        if s < warm:
            return s / max(1, warm)
        import math
        prog = (s - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)

    out_dir = os.path.join(ROOT, "outputs", "idx", run_name)
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
        mpath = os.path.join(out_dir, "metrics.csv")
        hdr = ("epoch,train_loss,mAP@0,mAP@1,mAP@2,mAP@4,mAP012,ndet,ngt\n" if multi_event
               else "epoch,train_loss,hit@0,hit@1,hit@2,hit@4,medAE,MAE,fail%\n")
        open(mpath, "w").write(hdr)
    best = -1.0
    gstep = 0
    for ep in range(epochs):
        W.model.train(); ad.train()
        rng = random.Random(1000 + ep)
        order = tr_idx[:]; rng.shuffle(order)          # same shuffle across ranks
        shard = order[rank::world]                     # disjoint per-rank shard
        run_loss, t0 = 0.0, time.time()
        for b0 in range(0, steps_per_epoch * batch_size, batch_size):
            batch = [train_ds[i] for i in shard[b0:b0 + batch_size]]
            loss = loz.loss_batch(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if distributed:                            # average the small trainable grads
                for p in trainable:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad /= world
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step(); gstep += 1
            run_loss += float(loss.detach())
            if is_main and gstep % 20 == 0:
                n = (b0 // batch_size) + 1
                print(f"  ep{ep} step{gstep} loss {run_loss/n:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{n*batch_size*world/(time.time()-t0):.1f} samp/s", flush=True)
        if is_main:
            avg = run_loss / max(1, steps_per_epoch)
            if multi_event:
                from eval.idx_metrics import evaluate_multievent_idx
                ap = evaluate_multievent_idx(loz, "val", FEAT, fps=fps, batch_size=batch_size,
                                             question_phase="test", types=types, max_videos=eval_videos)
                print(f"[val] ep{ep} loss {avg:.4f} | mAP@0 {ap['mAP@0']:.1f} @1 {ap['mAP@1']:.1f} "
                      f"@2 {ap['mAP@2']:.1f} @4 {ap['mAP@4']:.1f} | @012 {ap['mAP_012']:.1f} "
                      f"| ndet {ap['_avg_ndet']:.1f} vs ngt {ap['_avg_ngt']:.1f}", flush=True)
                open(mpath, "a").write(f"{ep},{avg:.4f},{ap['mAP@0']:.1f},{ap['mAP@1']:.1f},"
                                       f"{ap['mAP@2']:.1f},{ap['mAP@4']:.1f},{ap['mAP_012']:.1f},"
                                       f"{ap['_avg_ndet']:.1f},{ap['_avg_ngt']:.1f}\n")
                score = ap["mAP@2"]; ev = {"ap": ap}
            else:
                ev = evaluate(loz, val_ds, va_idx, batch_size)
                h = ev["hit"]
                print(f"[val] ep{ep} loss {avg:.4f} | hit@0 {h[0]:.1f} @1 {h[1]:.1f} @2 {h[2]:.1f} "
                      f"@4 {h[4]:.1f} | medAE {ev['medAE']:.1f} MAE {ev['MAE']:.1f} "
                      f"fail {ev['fail%']:.1f}% | clean {ev['clean%']:.0f}%", flush=True)
                print(f"      eg (gt,pred,nsame): {ev['examples']}", flush=True)
                open(mpath, "a").write(f"{ep},{avg:.4f},{h[0]:.1f},{h[1]:.1f},{h[2]:.1f},{h[4]:.1f},"
                                       f"{ev['medAE']:.1f},{ev['MAE']:.1f},{ev['fail%']:.1f}\n")
                score = h[2]
            if score > best:
                best = score
                save = {"adaptor": ad.state_dict(), "cfg": dict(use_idx=use_idx, lora=lora,
                        lora_rank=lora_rank, crop_secs=crop_secs, anchor_stride=anchor_stride,
                        multi_event=multi_event, types=list(types)), "epoch": ep, "ev": ev}
                if lora_params:
                    save["lora"] = [p.detach().cpu() for p in lora_params]
                torch.save(save, os.path.join(out_dir, "best.pt"))
                print(f"  [ckpt] best score {best:.1f} -> {out_dir}/best.pt", flush=True)
        if distributed:
            dist.barrier()
        loz.W.model.train()                            # restore after rank-0 eval
    if is_main:
        print(f"[done] {run_name} best {'mAP@2' if multi_event else 'hit@2'} {best:.1f}", flush=True)
    if distributed:
        dist.destroy_process_group()
    return best
