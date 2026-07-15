"""Phase 1 training loop — trains ONLY the V-JEPA adaptor + [LOC] embedding.

Frozen: Qwen3-VL (ViT+LLM) and V-JEPA. wandb logs live train/val loss, val MAE
(frames), and val AP@{0,1,2,4}. Large batch through the frozen 8B is affordable
because the forward is truncated to sim_layer and gradient-checkpointed.
"""
from __future__ import annotations

import math
import os
import sys
import time

import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, DistributedSampler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/home/chang_noroot/data2/hf_cache")

from data.dataset import Phase1Dataset, DetectionDataset, collate_list  # noqa: E402
from train.loss import (loc_loss, reject_loss, dist_contrastive_loss,   # noqa: E402
                        detection_loss)
from eval.metrics import evaluate                             # noqa: E402


def setup_dist():
    """Return (distributed, rank, world, local_rank, device)."""
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        return True, rank, world, local, f"cuda:{local}"
    return False, 0, 1, 0, "cuda"


def build(cfg, device="cuda"):
    from models.vjepa_adaptor import VJEPAAdaptor
    # ---- probe baseline (NO VLM): adaptor + learnable per-type query, no Qwen loaded ----
    if cfg.get("probe_mode", False):
        from models.probe import ProbeModel
        d_llm = 4096
        ad = VJEPAAdaptor(768, d_llm, hidden=cfg["adaptor_hidden"],
                          n_layers=cfg["adaptor_layers"]).to(device, torch.bfloat16)
        for p in ad.parameters():
            p.requires_grad_(True)
        probe = ProbeModel(ad, d_llm=d_llm, temp=cfg["temp"]).to(device)
        probe.query.requires_grad_(True)
        return None, ad, None, probe

    from models.wrapper import QwenWrapper
    from models.loc_tokens import LocTokens
    from models.localizer import Localizer
    from models.sim_head import SimHead

    W = QwenWrapper(device=device, dtype=torch.bfloat16)
    ad = VJEPAAdaptor(768, W.d_llm, hidden=cfg["adaptor_hidden"],
                      n_layers=cfg["adaptor_layers"]).to(device, torch.bfloat16)
    ad.set_target_rms_from(W.embed_tokens.weight)
    loc = LocTokens(W.d_llm, k=1).to(device, torch.bfloat16)
    loc.init_from_embeddings(W.embed_tokens.weight)
    head = None
    if cfg.get("sim_head", True):
        head = SimHead(W.d_llm, d_proj=cfg.get("sim_proj", 256)).to(device)  # fp32 head
    film = None
    if cfg.get("film", False):
        from models.film import FiLM
        film = FiLM(W.d_llm, hidden=cfg.get("film_hidden", 1024)).to(device)  # fp32
    for p in ad.parameters():
        p.requires_grad_(True)
    loc.loc.requires_grad_(True)
    loz = Localizer(W, ad, loc, cfg["sim_layer"], temp=cfg["temp"],
                    use_anchor=cfg["use_anchor"], fps=cfg["fps"],
                    grad_checkpoint=cfg["grad_checkpoint"], sim_head=head,
                    align_target=cfg.get("align_target", "postllm"), film=film)
    loz.anchor_max_side = cfg.get("anchor_max_side", 0)
    loz.lora_params = []
    if cfg.get("lora", False):
        loz.lora_params = W.add_lora(rank=cfg.get("lora_rank", 16), alpha=cfg.get("lora_alpha", 32),
                                     n_layers=cfg["sim_layer"], target=cfg.get("lora_target", "all"))
    return W, ad, loc, loz


def trainable_params(ad, loc, loz):
    if loz.__class__.__name__ == "ProbeModel":
        return list(loz.parameters())        # adaptor + per-type query
    ps = list(ad.parameters()) + [loc.loc]
    if loz.sim_head is not None:
        ps += list(loz.sim_head.parameters())
    if loz.film is not None:
        ps += list(loz.film.parameters())
    ps += getattr(loz, "lora_params", [])
    return ps


def trainable_report(ad, loc, loz, W):
    tp = sum(p.numel() for p in trainable_params(ad, loc, loz))
    if W is None:                            # probe: no VLM
        print(f"[params] PROBE trainable {tp/1e6:.2f}M (no VLM loaded)")
        return tp
    total = tp + sum(p.numel() for p in W.model.parameters())
    frac = tp / total
    print(f"[params] trainable {tp/1e6:.2f}M / total {total/1e9:.2f}B = {frac*100:.4f}%")
    assert frac < 0.05, "trainable params exceed 5% — backbone not frozen?"
    return tp


def run(cfg_path=None, **over):
    cfg = yaml.safe_load(open(cfg_path or os.path.join(ROOT, "configs", "train.yaml")))
    cfg.update(over)
    distributed, rank, world, local, device = setup_dist()
    is_main = rank == 0
    torch.manual_seed(0)                       # same init across ranks (no DDP wrap)

    W, ad, loc, loz = build(cfg, device)
    if is_main:
        trainable_report(ad, loc, loz, W)
    trainable = trainable_params(ad, loc, loz)

    detection = cfg.get("task_mode", "single") == "detection"
    train_types = tuple(cfg.get("train_types", ["touch", "untouch"]))
    train_cats = cfg.get("train_categories")            # None = all
    excl_cats = cfg.get("exclude_categories")           # held-out for generalization
    if detection:
        train_ds = DetectionDataset("train", cfg["feat_dir"], types=train_types,
                                    crop_secs=cfg.get("crop_secs", 0), p_neg=cfg.get("p_neg", 0.2),
                                    question_phase="train", categories=train_cats,
                                    exclude_categories=excl_cats)
        val_ds = None                              # val uses evaluate_multievent (full video)
    else:
        train_ds = Phase1Dataset("train", cfg["feat_dir"], crop_secs=cfg.get("crop_secs", 0),
                                 p_neg=cfg.get("p_neg", 0.0))
        val_ds = Phase1Dataset("val", cfg["feat_dir"], crop_secs=0)
    if is_main:
        vn = "multi-event(full-video)" if detection else len(val_ds)
        print(f"[data] train {len(train_ds)} | val {vn} | world={world} "
              f"| per-gpu batch {cfg['batch_size']} | eff batch {cfg['batch_size']*world} "
              f"| task={cfg.get('task_mode','single')}")
    sampler = (DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True,
                                  drop_last=True) if distributed else None)
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=(sampler is None),
                        sampler=sampler, collate_fn=collate_list,
                        num_workers=cfg.get("num_workers", 4), drop_last=True)

    if loz.__class__.__name__ == "ProbeModel":
        groups = [{"params": list(ad.parameters()), "lr": cfg["lr_adaptor"]},
                  {"params": [loz.query], "lr": cfg["lr_loc"]}]
    else:
        groups = [
            {"params": list(ad.parameters()), "lr": cfg["lr_adaptor"]},
            {"params": [loc.loc], "lr": cfg["lr_loc"]},
        ]
        if loz.sim_head is not None:
            groups.append({"params": list(loz.sim_head.parameters()), "lr": cfg["lr_adaptor"]})
        if loz.film is not None:
            groups.append({"params": list(loz.film.parameters()), "lr": cfg["lr_adaptor"]})
        if getattr(loz, "lora_params", []):
            groups.append({"params": loz.lora_params, "lr": cfg.get("lr_lora", 2.0e-4)})
    opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    steps_per_epoch = len(loader) // cfg["grad_accum"]
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup = int(total_steps * cfg["warmup_frac"])

    def lr_at(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    wb = None
    if is_main and cfg.get("wandb", True):
        try:
            import wandb
            wb = wandb.init(project=cfg["wandb_project"], config=cfg,
                            name=cfg.get("run_name", f"phase1_L{cfg['sim_layer']}"))
            print(f"[wandb] {wb.url}")
        except Exception as e:
            print(f"[wandb] disabled ({e})")

    out_dir = os.path.join(ROOT, "outputs", "phase1", cfg.get("run_name", "run"))
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, "metrics.csv")      # LOCAL log (no wandb needed)
    if is_main and not os.path.exists(metrics_path):
        open(metrics_path, "w").write("epoch,gstep,train_loss,mAP@0,mAP@1,mAP@2,mAP@4,mAP@012\n")
    best = -1.0
    gstep = 0
    for epoch in range(cfg["epochs"]):
        if sampler is not None:
            sampler.set_epoch(epoch)
        ad.train(); loz.train()
        opt.zero_grad(set_to_none=True)
        t0 = time.time()
        run_loss = 0.0
        for it, batch in enumerate(loader):
            s_list, metas = loz.forward_batch(batch)
            loss = 0.0
            for s, m in zip(s_list, metas):
                if detection:                         # multi-event per-frame detection (empty=all-neg)
                    l = detection_loss(s, m.get("event_frames") or [],
                                       dilate=cfg.get("det_dilate", 2),
                                       pos_weight=cfg.get("det_pos_weight", 15.0))
                elif m["gt"] < 0:                     # negative window -> scale-free peak reject
                    l = cfg.get("lambda_neg", 1.0) * reject_loss(s, cfg.get("rej_peak_mult", 3.0))
                elif cfg.get("loss_mode", "ce") == "dist":
                    l, _ = dist_contrastive_loss(
                        s, m["gt"], sigma=cfg.get("dist_sigma", 4.0),
                        far_margin=cfg.get("dist_far_margin", 5.0),
                        far_weight=cfg.get("dist_far_weight", 1.0),
                        lambda_mae=cfg["lambda_mae"])
                else:
                    l, _ = loc_loss(s, m["gt"], lambda_mae=cfg["lambda_mae"],
                                    label_smooth_frames=cfg["label_smooth_frames"],
                                    neighbor_k=cfg["neighbor_k"])
                loss = loss + l
            loss = loss / len(s_list) / cfg["grad_accum"]
            loss.backward()
            run_loss += float(loss.detach()) * cfg["grad_accum"]
            if (it + 1) % cfg["grad_accum"] == 0:
                if distributed:                # average the 10M trainable grads across GPUs
                    for p in trainable:
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                            p.grad /= world
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                gstep += 1
                if is_main and gstep % 20 == 0:
                    avg = run_loss / (it + 1)
                    print(f"  ep{epoch} step{gstep} loss {avg:.4f} "
                          f"lr {sched.get_last_lr()[0]:.2e} "
                          f"{(it+1)*world/(time.time()-t0):.1f} samp*it/s", flush=True)
                    if wb:
                        wb.log({"train/loss": avg, "train/lr": sched.get_last_lr()[0],
                                "epoch": epoch}, step=gstep)
        # ---- validation (rank 0) ----
        if (epoch + 1) % cfg["eval_every"] == 0 or epoch == cfg["epochs"] - 1:
            if is_main:
                if detection:
                    from eval.metrics import evaluate_multievent
                    ap = evaluate_multievent(
                        loz, "val", cfg["feat_dir"], fps=cfg["fps"],
                        window_secs=cfg.get("crop_secs", 10),
                        stride_secs=max(1, cfg.get("crop_secs", 10) // 2),
                        tolerances=cfg["tolerances"], batch_size=cfg["batch_size"],
                        min_gap=cfg.get("nms_min_gap", 15), rel_thresh=cfg.get("nms_rel_thresh", 0.5),
                        categories=train_cats, exclude_categories=excl_cats, types=train_types)
                    val = {"ap": ap, "loss": 0.0, "mae": {"all": 0.0}}
                    print(f"[val] ep{epoch} MULTI-EVENT mAP@2 {ap['mAP@2']:.2f} "
                          f"mAP@012 {ap['mAP_012']:.2f} mAP@4 {ap['mAP@4']:.2f}", flush=True)
                else:
                    val = eval_split(loz, val_ds, cfg)
                    print(f"[val] ep{epoch} loss {val['loss']:.4f} MAE {val['mae']['all']:.2f} "
                          f"mAP@2 {val['ap']['mAP@2']:.2f} mAP@012 {val['ap']['mAP_012']:.2f}", flush=True)
                if wb:
                    wb.log({**{f"val/{k}": v for k, v in val["ap"].items()}, "epoch": epoch}, step=gstep)
                a = val["ap"]
                open(metrics_path, "a").write(
                    f"{epoch},{gstep},{run_loss/max(1,len(loader)):.4f},{a.get('mAP@0',0):.2f},"
                    f"{a.get('mAP@1',0):.2f},{a['mAP@2']:.2f},{a.get('mAP@4',0):.2f},{a['mAP_012']:.2f}\n")
                score = val["ap"]["mAP@2"]
                if score > best:
                    best = score
                    if loz.__class__.__name__ == "ProbeModel":
                        save = {"probe": loz.state_dict(), "adaptor": ad.state_dict(),
                                "cfg": cfg, "epoch": epoch, "val": val}
                    else:
                        save = {"adaptor": ad.state_dict(), "loc": loc.state_dict(),
                                "cfg": cfg, "epoch": epoch, "val": val}
                        if loz.sim_head is not None:
                            save["sim_head"] = loz.sim_head.state_dict()
                        if loz.film is not None:
                            save["film"] = loz.film.state_dict()
                        if getattr(loz, "lora_params", []):
                            save["lora"] = [p.detach().cpu() for p in loz.lora_params]
                    torch.save(save, os.path.join(out_dir, "best.pt"))
                    print(f"  [ckpt] new best mAP@2 {best:.2f} -> {out_dir}/best.pt")
            if distributed:
                dist.barrier()
    if wb:
        wb.finish()
    if distributed:
        dist.destroy_process_group()
    if is_main:
        print(f"[done] best val mAP@2 {best:.2f}")
    return best


@torch.no_grad()
def eval_split(loz, ds, cfg):
    """Sliding-window validation (matches the cropped training regime) -> AP/MAE.
    'loss' here is just the mean MAE (frames) so the wandb curve is meaningful."""
    import numpy as np
    from collections import defaultdict
    from eval.metrics import predict_windowed, build_truth_pred
    from common.score import compute_mAPs

    win = cfg.get("crop_secs", 10) or 10
    preds = predict_windowed(loz, ds, window_secs=win, stride_secs=max(1, win // 2),
                             fps=cfg["fps"], batch_size=cfg["batch_size"])
    err = defaultdict(list)
    for p in preds:
        e = abs(p["pred_frame"] - p["gt"])
        err["all"].append(e); err[p["type"]].append(e)
    truth, pred = build_truth_pred(preds)
    mAPs, _ = compute_mAPs(truth, pred, tolerances=cfg["tolerances"])
    ap = {f"mAP@{t}": float(mAPs[i]) * 100 for i, t in enumerate(cfg["tolerances"])}
    ap["mAP_012"] = float(np.mean([mAPs[i] for i, t in enumerate(cfg["tolerances"])
                                   if t in (0, 1, 2)])) * 100
    mae = {k: float(np.mean(v)) for k, v in err.items()}
    tot, n = mae["all"], 1
    return {"loss": tot / max(n, 1), "mae": mae, "ap": ap}


if __name__ == "__main__":
    run()
