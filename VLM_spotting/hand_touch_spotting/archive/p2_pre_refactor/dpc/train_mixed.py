#!/usr/bin/env python3
"""Mixed multi-dataset training on DPC for the idx-compress spotter, with ON-THE-FLY V-JEPA.

DATA-PARALLEL (DDP via torchrun): each rank streams a disjoint shard of the windows,
runs its own frozen Qwen + V-JEPA, and the small trainable FrameCompress (+optional LoRA)
gradients are all-reduced every step. rank 0 owns eval / wandb / checkpoint.

Pipeline (no precomputed features on disk):
  tar frames (TarFrameSource) --window--> uint8 (T,H,W,3)
    --frozen OnlineVJEPA--> grid (ceil(T/2),576,768) fp16   [in the loop, no_grad]
    --FrameCompress(question-conditioned)--> N motion tokens
    --frozen Qwen3-VL-8B (idx-decode)--> generate frame indices
Trainable = FrameCompress (+ optional LoRA). Frozen = V-JEPA + Qwen.

In-domain (trained + in-domain val): TouchMoment/HOI4D+TACO, tennis, finegym, soccernet_ball, fs_perf.
Out-of-domain (NEVER trained, zero-shot val): finediving.
Eval is per-WINDOW mAP (each window = a clip) — a faithful spotting proxy for the pilot.
"""
from __future__ import annotations
import argparse, json, math, os, random, sys, time
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from dpc import paths                                                  # noqa: E402
os.environ.setdefault("HF_HOME", paths.HF_HOME)
os.environ.setdefault("VJEPA_REPO", paths.VJEPA_REPO)
os.environ.setdefault("VJEPA_CKPT", paths.VJEPA_CKPT)

from dpc.eval_vendor import eval_nms as en                            # noqa: E402
from dpc.frame_source import TarFrameSource, DirFrameSource            # noqa: E402
from dpc.windowed_dataset import WindowedSpottingDataset              # noqa: E402
from dpc.vjepa_online import OnlineVJEPA                              # noqa: E402

IN_DOMAIN = ["touchmoment", "tennis", "finegym", "fs_perf"]   # touchmoment = merged hoi4d+taco; fs_perf == fs_comp data; soccernet dropped (Phase 2)
OOD = ["finediving"]


def load_key_split(ann_dir, key, split):
    p = os.path.join(ann_dir, key, f"{split}.json")
    return json.load(open(p)) if os.path.exists(p) else []


def combined(ann_dir, keys, split):
    out = []
    for k in keys:
        out += load_key_split(ann_dir, k, split)
    return out


def _prefetch_batches(ds, order, bs, pool, ahead):
    """Yield lists of `bs` window dicts (ds[order[i]]), prefetching the frame-decode of up to
    `ahead` batches in background `pool` threads so decode (CPU, GIL-releasing) overlaps the
    GPU forward/backward. Order preserved. Frame source must be thread-safe (TarFrameSource is,
    via per-thread stores)."""
    from collections import deque
    idxs = order[:(len(order) // bs) * bs]
    futs = deque(); it = iter(idxs)
    for _ in range(max(1, ahead) * bs):                    # prime the pipeline
        try:
            futs.append(pool.submit(ds.__getitem__, next(it)))
        except StopIteration:
            break
    cur = []
    while futs:
        w = futs.popleft().result()
        try:
            futs.append(pool.submit(ds.__getitem__, next(it)))
        except StopIteration:
            pass
        cur.append(w)
        if len(cur) == bs:
            yield cur; cur = []


def make_dataset(records, fs, a, phase, balance=None, tag="ds", jitter=0, cross_neg=0.0, temp_alpha=0.0,
                 type2_rate=0.0, neg_cap=0.40, neg_cap_by_ds=None):
    tmp = os.path.join(a.out, f"_ann_{tag}.json")
    os.makedirs(a.out, exist_ok=True)
    json.dump(records, open(tmp, "w"))
    return WindowedSpottingDataset(tmp, fs, window_frames=a.window_frames, stride=a.stride,
                                   question_phase=phase, negative_rate=a.negative_rate,
                                   balance_per_dataset=balance, seed=a.seed, jitter=jitter,
                                   cross_neg_rate=cross_neg, temp_alpha=temp_alpha,
                                   type2_rate=type2_rate, neg_cap=neg_cap, neg_cap_by_ds=neg_cap_by_ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--out", default=os.path.join(paths.OUT_DIR, "mixed"))
    ap.add_argument("--balance", type=int, default=2500, help="max (window,type) samples per dataset (domain-balanced)")
    ap.add_argument("--eval_balance", type=int, default=30, help="eval (window,type) samples per dataset (balanced)")
    ap.add_argument("--eval_balance_finegym", type=int, default=-1, help="override eval_balance for finegym (32 types need more); <0 = use eval_balance")
    ap.add_argument("--eval_balance_ood", type=int, default=-1, help="override eval_balance for the OOD set (finediving); <0 = use eval_balance. Set high (e.g. 1018=full val) for a stable OOD number")
    ap.add_argument("--window_frames", type=int, default=600, help="max frames per window (clip cap)")
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--negative_rate", type=float, default=0.15)
    ap.add_argument("--cross_neg_rate", type=float, default=0.0, help="per-window prob of a CROSS-DATASET hard negative (foreign action -> NL 'no frame related'); train only")
    ap.add_argument("--temp_alpha", type=float, default=0.0, help="dataset-level temperature exponent for shuffle-epoch draws ~n^alpha; 0.75 gives finegym >=1 full pass + small sets 1.5-2x; 0=natural; train only")
    ap.add_argument("--type2_rate", type=float, default=0.0, help="in-domain hard negatives per window: ask ANOTHER same-dataset type absent from the clip -> NL 'no frame'; >1 fills trimmed sets toward neg_cap; train only")
    ap.add_argument("--neg_cap", type=float, default=0.40, help="cap negatives (type1/2/3) at this fraction of a dataset's samples (drop random excess) so the model doesn't over-predict 'none'; train only")
    ap.add_argument("--neg_cap_finegym", type=float, default=-1.0, help="override neg_cap for finegym only (<0 = use global neg_cap); lower (e.g. 0.20) shrinks finegym's pool so its positives get full coverage at lower alpha + less 'none' bias from the dominant domain")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8, help="micro-steps per optimizer step; "
                    "effective batch = batch_size x world_size x grad_accum")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=20, help="linear LR warmup (opt steps)")
    ap.add_argument("--lr_lora", type=float, default=1e-4)
    ap.add_argument("--lora", type=int, default=0)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--n_q", type=int, default=8)
    ap.add_argument("--stain", type=int, default=0)
    ap.add_argument("--use_text", type=int, default=0)
    ap.add_argument("--gate_lang", type=int, default=0)
    ap.add_argument("--use_anchor", type=int, default=1, help="frozen Qwen-ViT anchors (base was trained WITH them)")
    ap.add_argument("--anchor_stride", type=int, default=5, help="place a ViT anchor every N seconds")
    ap.add_argument("--anchor_max_side", type=int, default=252)
    ap.add_argument("--jitter", type=int, default=0, help="train-only window-start jitter (frames) to break position prior; 0=off")
    ap.add_argument("--vjepa_bf16", type=int, default=0, help="run V-JEPA encoder under autocast(bf16): ~2x faster, RoPE stays fp32")
    ap.add_argument("--fsdp", type=int, default=0, help="FSDP-shard the frozen Qwen decoder layers across ranks (frees ~12GB/GPU for window600+anchors)")
    ap.add_argument("--ckpt_every", type=int, default=50, help="save resume checkpoint every N opt-steps (preempt-safe); also always saves at each eval")
    ap.add_argument("--ckpt_secs", type=int, default=150, help="ALSO save resume every N seconds (survives frequent Kueue preemption regardless of step speed)")
    ap.add_argument("--eval_max_tokens", type=int, default=24, help="max generated tokens per eval window (~8 events); lower = fewer per-token FSDP all-gathers = faster eval")
    ap.add_argument("--tols", default="0,1,2,4", help="comma-sep frame tolerances for mAP, e.g. '0,2,4,8,15' (15 frames ~= 0.6s @25fps)")
    ap.add_argument("--eval_only", type=int, default=0, help="load --init_fc, run one eval, exit (run with --fsdp 0 for fast 4-GPU independent full-model eval); no training")
    ap.add_argument("--diag_rollout", type=int, default=0, help="GRPO rollout-diversity diagnosis: sample G rollouts per window on N positive windows, report within-group reward(F1@2) std (GRPO needs std>0). Requires --eval_only 1. 0=off")
    ap.add_argument("--diag_G", type=int, default=8, help="rollouts per window for --diag_rollout")
    ap.add_argument("--diag_temps", default="0.7,1.0,1.3", help="comma-sep sampling temperatures to sweep for --diag_rollout")
    ap.add_argument("--init_fc", default="")
    ap.add_argument("--eval_windows", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--cpu_threads", type=int, default=6)
    ap.add_argument("--num_workers", type=int, default=4, help="background threads prefetching frame decode (overlaps CPU decode with GPU compute); 0=synchronous")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb_project", default="dpc_mixed_spotting")
    ap.add_argument("--wandb_mode", default="online")
    ap.add_argument("--local_frames", default="")
    a = ap.parse_args()
    if a.stride is None:
        a.stride = a.window_frames
    torch.manual_seed(a.seed); random.seed(a.seed); torch.set_num_threads(a.cpu_threads)
    en.TOLS = [int(x) for x in str(a.tols).split(",") if x.strip() != ""]   # frame tolerances for mAP

    # ---------------- DDP (torchrun) ----------------
    LR = int(os.environ.get("LOCAL_RANK", -1))
    ddp = LR >= 0
    if ddp:
        torch.cuda.set_device(LR)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{LR}"),
                                timeout=timedelta(minutes=30))
        rank, world, dev = dist.get_rank(), dist.get_world_size(), f"cuda:{LR}"
    else:
        rank, world, dev = 0, 1, "cuda"
    is_main = rank == 0

    def log0(*x):
        if is_main:
            print(*x, flush=True)

    if is_main:
        os.makedirs(a.out, exist_ok=True)

    # ---------------- data ----------------
    fs = (DirFrameSource(a.local_frames) if a.local_frames
          else TarFrameSource(paths.DATA_ROOT, paths.INDEX))
    train_ds = make_dataset(combined(paths.ANN_DIR, IN_DOMAIN, "train"), fs, a, "train",
                            balance=a.balance, tag=f"train_r{rank}", jitter=a.jitter,
                            cross_neg=a.cross_neg_rate, temp_alpha=a.temp_alpha,
                            type2_rate=a.type2_rate, neg_cap=a.neg_cap,
                            neg_cap_by_ds=({"finegym": a.neg_cap_finegym} if a.neg_cap_finegym >= 0 else None))
    in_bal = ({"finegym": a.eval_balance_finegym, "default": a.eval_balance}
              if a.eval_balance_finegym >= 0 else a.eval_balance)   # finegym (32 types) can get more
    ood_bal = a.eval_balance_ood if a.eval_balance_ood >= 0 else a.eval_balance
    inval_ds = make_dataset(combined(paths.ANN_DIR, IN_DOMAIN, "val"), fs, a, "test",
                            balance=in_bal, tag=f"inval_r{rank}")
    ood_ds = make_dataset(combined(paths.ANN_DIR, OOD, "val"), fs, a, "test",
                          balance=ood_bal, tag=f"ood_r{rank}")
    log0(f"[data] world={world} train windows={len(train_ds)} in-val={len(inval_ds)} ood-val={len(ood_ds)}")

    # ---------------- model ----------------
    from models.wrapper import QwenWrapper
    from models.vjepa_adaptor import VJEPAAdaptor
    from models.frame_compress import FrameCompress
    from models.idx_localizer import IdxLocalizer

    qwen_path = os.environ.get("QWEN_PATH", paths.QWEN_PATH)
    W = QwenWrapper(model_id=qwen_path, device=dev, dtype=torch.bfloat16)
    ad = VJEPAAdaptor(768, W.d_llm, hidden=2048, n_layers=2).to(dev, torch.bfloat16)
    ad.set_target_rms_from(W.embed_tokens.weight)
    fc = FrameCompress(768, a.n_q, W.d_llm, n_heads=4, stain=bool(a.stain),
                       use_text=bool(a.use_text), gate_lang=bool(a.gate_lang)).to(dev, torch.float32)
    fc.set_target_rms_from(W.embed_tokens.weight)
    if a.init_fc and os.path.exists(a.init_fc):
        sd = torch.load(a.init_fc, weights_only=False, map_location="cpu")["fc"]
        miss, unexp = fc.load_state_dict(sd, strict=False)
        log0(f"[init] loaded base fc {a.init_fc} (missing {len(miss)} unexpected {len(unexp)})")
    trainable = list(fc.parameters()); front = trainable; lora_params = []
    if a.lora:
        lora_params = W.add_lora(rank=a.lora_rank, alpha=2 * a.lora_rank, n_layers=W.n_layers, target="all")
        trainable = front + lora_params
    loz = IdxLocalizer(W, ad, use_idx=True, use_anchor=bool(a.use_anchor),
                       anchor_stride=a.anchor_stride, anchor_max_side=a.anchor_max_side, use_mrope=False,
                       fps=15, max_frames=a.window_frames, grad_checkpoint=True, compress=fc)
    vjepa = OnlineVJEPA(device=dev, dtype=torch.float16,
                        compute_dtype=(torch.bfloat16 if a.vjepa_bf16 else torch.float32))
    log0(f"[cfg] trainable={sum(p.numel() for p in trainable)/1e6:.2f}M window={a.window_frames} "
         f"bs={a.batch_size} accum={a.grad_accum} eff_batch={a.batch_size*world*a.grad_accum} balance={a.balance}/ds")

    # -------- optional: FSDP-shard the FROZEN Qwen decoder layers across ranks --------
    # 16GB bf16 weights -> ~4GB/GPU sharded, freeing ~12GB so window600+anchors *backward*
    # fits on 40GB. Only Qwen3VLTextDecoderLayer units are sharded; embed_tokens (read
    # directly by the connector) and the ViT (called directly for anchors) are kept FULL.
    # Base is frozen -> FSDP still all-gathers each layer in BACKWARD (triggered by the loss
    # output requiring grad, torch>=2.1 PR#101982), so input-grads reach the trainable
    # connector; no LLM param-grads are reduced. Connector stays replicated + manually
    # all-reduced (below). No MixedPrecision: params are already bf16, nothing to reduce.
    fsdp_handle = None                            # set to the FSDP root when --fsdp (used to summon full params in eval)
    if ddp and a.fsdp:
        import functools
        from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, ShardingStrategy,
                                            BackwardPrefetch)
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer
        wrap_pol = functools.partial(transformer_auto_wrap_policy,
                                     transformer_layer_cls={Qwen3VLTextDecoderLayer})
        # forward_prefetch + no limit_all_gathers: overlap each layer's all-gather with the previous
        # layer's compute so the ~130ms/all-gather latency (NV4, no NVSwitch) is hidden -> much faster
        # autoregressive eval (and training). backward_prefetch same idea for the bwd pass.
        fsdp_llm = FSDP(W.llm, auto_wrap_policy=wrap_pol,
                        sharding_strategy=ShardingStrategy.FULL_SHARD,
                        device_id=torch.cuda.current_device(),
                        use_orig_params=True, sync_module_states=False,
                        limit_all_gathers=False, forward_prefetch=True,
                        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                        ignored_modules=[W.embed_tokens])
        W.llm = fsdp_llm
        W.core.language_model = fsdp_llm          # W.model(...) forward now hits the sharded llm
        fsdp_handle = fsdp_llm
        torch.cuda.empty_cache()
        log0(f"[fsdp] FULL_SHARD Qwen decoder layers across {world} ranks; embed_tokens + ViT kept full")

    def to_sample(ws):
        N = int(ws["num_win_frames"]); fps_i = max(1, int(round(ws["fps"] or 15)))
        s = {"question": ws["question"], "fps": fps_i, "num_frames": N,
             "video_id": f"{ws['video_id']}#w{ws['win_start']}", "type": ws["type"],
             "dataset": ws.get("dataset"),
             "event_frames": [int(x) for x in ws["target_frames"]], "gt": -1,
             "anchor_start_sec": 0, "anchor_num_secs": math.ceil(N / fps_i),
             "frames": ws["frames"]}                          # window frames -> ViT anchors (window-local)
        s["grid"] = vjepa.extract_grid(ws["frames"])
        return s

    @torch.no_grad()
    def evaluate(ds, tag):
        """Parallel eval: each rank generates on its SHARD of the windows, then partial
        (FSDP autoregressive eval is slow/fragile on no-NVSwitch nodes -> set --eval_windows 0
        for loss-only training and eval the saved connector separately on 1 GPU.)"""
        if a.eval_windows <= 0:
            return None
        loz.W.model.eval(); fc.eval()
        # FSDP note: sharded weights make EVERY generate-forward a collective all-gather, so all
        # ranks must stay in lockstep. Independent per-rank shards desync (a fast rank reaches
        # all_gather_object while a slow rank is still gathering weights -> NCCL timeout/abort). So
        # under FSDP we run eval REPLICATED: every rank generates the SAME windows (identical greedy
        # forwards -> identical all-gather sequence -> lockstep) and only rank0 RECORDS; the other
        # ranks generate purely to stay in lockstep. Without FSDP we keep the cheap sharded eval.
        if fsdp_handle is not None:
            idxs = list(range(min(len(ds), a.eval_windows)))              # replicated (FSDP lockstep)
        else:
            idxs = list(range(min(len(ds), a.eval_windows)))[rank::world] # sharded (fast, no FSDP)
        # (summon_full_params does NOT help: FULL_SHARD reshards after every forward, so each
        # generated token re-all-gathers regardless. forward_prefetch=True on the FSDP module is
        # what actually hides the all-gather latency. Eval stays replicated for lockstep.)
        truth, pred, dsof = {}, {}, {}
        _n0 = time.time(); _done = 0
        for b0 in range(0, len(idxs), a.batch_size):
            samples = [to_sample(ds[i]) for i in idxs[b0:b0 + a.batch_size]]
            for s, (fr, sc) in zip(samples, loz.predict_multievent_batch(samples, max_new_tokens=a.eval_max_tokens)):
                _done += 1
                if is_main and _done % 20 == 0:
                    log0(f"  [eval {tag}] {_done}/{len(idxs)} win  {(time.time()-_n0)/_done:.2f}s/win")
                if fsdp_handle is not None and not is_main:
                    continue                                             # non-main: generate for lockstep, don't record
                vid, t, N = s["video_id"], s["type"], s["num_frames"]
                truth.setdefault(vid, []); pred.setdefault(vid, []); dsof[vid] = s.get("dataset", "?")
                truth[vid] += [{"label": t, "frame": int(f)} for f in s["event_frames"]]
                pred[vid] += [{"label": t, "frame": int(f), "score": float(p)}
                              for f, p in zip(fr, sc) if 0 <= f < N]
        if ddp:
            parts = [None] * world
            dist.all_gather_object(parts, (truth, pred, dsof))   # collective sync (all ranks work)
        else:
            parts = [(truth, pred, dsof)]
        if not is_main:
            return None
        TR, PR, DS = {}, {}, {}
        for t_, p_, d_ in parts:
            for v, e in t_.items():
                TR.setdefault(v, []).extend(e)
            for v, e in p_.items():
                PR.setdefault(v, []).extend(e)
            DS.update(d_)
        groups = {}                                          # dataset -> [video_ids]
        for v in TR:
            groups.setdefault(DS.get(v, "?"), []).append(v)

        def score(vs):
            nd = sum(len(PR.get(v, [])) for v in vs)
            ng = sum(len(TR[v]) for v in vs)
            if ng == 0:                                       # no GT in this group -> scorer returns nan
                return [0.0] * len(en.TOLS), nd, ng
            try:
                m = en.maps_quiet([{"video": v, "events": TR[v]} for v in vs],
                                  [{"video": v, "events": PR.get(v, [])} for v in vs])
                m = [0.0 if (x != x) else float(x) for x in m]    # nan -> 0
            except Exception as e:                            # never let one dataset kill eval
                log0(f"  [eval warn] score failed ({e}); zeros for this group")
                m = [0.0] * len(en.TOLS)
            return m, nd, ng

        results = {"_all": score(list(TR.keys()))}           # overall + per-dataset
        for dsk in sorted(groups):
            results[dsk] = score(groups[dsk])
        for k in ["_all"] + sorted(groups):
            m, nd, ng = results[k]
            log0(f"[eval:{tag}:{k:18s}] mAP@{tuple(en.TOLS)}={[round(x, 2) for x in m]} ndet={nd} ngt={ng}")
        return results

    # ---------------- optim + wandb (rank 0) ----------------
    groups = [{"params": front, "lr": a.lr}]
    if lora_params:
        groups.append({"params": lora_params, "lr": a.lr_lora})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    per_rank = max(1, ((len(train_ds) // world) // a.batch_size) // a.grad_accum)
    total_steps = max(1, a.max_steps if a.max_steps else per_rank * a.epochs)

    def lr_lambda(step):                                  # linear warmup -> cosine decay
        if step < a.warmup:
            return (step + 1) / max(1, a.warmup)
        prog = (step - a.warmup) / max(1, total_steps - a.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    wb = None
    csvp = os.path.join(a.out, f"{a.run_name}_metrics.csv")
    if is_main:
        open(csvp, "w").write("step,split,dataset," + ",".join(f"mAP{t}" for t in en.TOLS) + ",ndet,ngt\n")
        if a.wandb_mode != "disabled":
            try:                                          # local runs can have a shadowing ./wandb dir
                import wandb
                if hasattr(wandb, "init"):
                    wb = wandb
                    wb.init(project=a.wandb_project, name=a.run_name, mode=a.wandb_mode,
                            config=vars(a), dir=a.out)
                else:
                    log0("[wandb] shadowed import (no .init) -> skipping wandb, mAP still goes to csv/console")
            except Exception as e:
                log0(f"[wandb] disabled ({e}); mAP still goes to csv/console")

    best = {"v": -1.0}

    def run_eval(step):
        for ds, tag in [(inval_ds, "in"), (ood_ds, "ood")]:
            res = evaluate(ds, tag)                          # dict {'_all':(m,nd,ng), '<ds>':...} on rank0
            if is_main and res is not None:
                for k, (m, nd, ng) in res.items():
                    pref = f"val/{tag}" if k == "_all" else f"val/{tag}/{k}"
                    if wb is not None:
                        d = {f"{pref}_mAP@{t}": m[i] for i, t in enumerate(en.TOLS)}
                        d[f"{pref}_ndet"] = nd; d[f"{pref}_ngt"] = ng
                        wb.log(d, step=step)
                    open(csvp, "a").write(f"{step},{tag},{k}," + ",".join(f"{x:.2f}" for x in m) + f",{nd},{ng}\n")
                _i2 = en.TOLS.index(2) if 2 in en.TOLS else min(2, len(en.TOLS) - 1)
                m2 = res["_all"][0][_i2]                      # overall in-domain mAP@2 -> ckpt selection
                if tag == "in" and m2 > best["v"]:
                    best["v"] = m2
                    ck = {"fc": fc.state_dict(), "m_in2": m2, "step": step, "cfg": vars(a)}
                    if lora_params:
                        ck["lora"] = [p.detach().cpu() for p in lora_params]
                    torch.save(ck, os.path.join(a.out, f"{a.run_name}_best.pt"))
                    log0(f"  [ckpt] best in-mAP@2={best['v']:.2f} @step{step}")
        fc.train(); loz.W.model.eval()                       # all ranks resume training mode

    def save_conn(step):
        """Save a NAMED connector checkpoint every eval (all kept, never overwritten) so the
        OOD-optimal step can be selected offline. Phase-1 lost its step-600 OOD peak to a single
        overwritten resume file; this prevents that. Small (~27.5M) -> cheap to keep ~10-15 of."""
        if not is_main:
            return
        ck = {"fc": fc.state_dict(), "gstep": step, "cfg": vars(a)}
        if lora_params:
            ck["lora"] = [p.detach().cpu() for p in lora_params]
        torch.save(ck, os.path.join(a.out, f"{a.run_name}_conn_s{step}.pt"))
        log0(f"  [ckpt] saved connector {a.run_name}_conn_s{step}.pt")

    # ---- eval-only: eval the --init_fc connector once and exit (no training) ----
    # Run with --fsdp 0 so each of the 4 ranks holds a FULL model and evaluates its OWN window
    # shard independently (no FSDP collectives during generation -> fast). This is the decoupled
    # eval for FSDP-trained checkpoints ("4 GPUs each as an independent single-GPU eval").
    if a.eval_only:
        step0 = 0                                          # log mAP at the checkpoint's TRAIN step
        if a.init_fc and os.path.exists(a.init_fc):
            try:
                _ck = torch.load(a.init_fc, map_location="cpu", weights_only=False)
                step0 = int(_ck.get("gstep", _ck.get("step", 0)))
            except Exception:
                pass
        if a.diag_rollout > 0:                             # GRPO rollout-diversity diagnosis (no training/eval)
            import numpy as _np
            def _f1(pred, gt, tol):                        # point-event F1 with greedy match within tol
                gt = list(gt); matched = [False] * len(gt); tp = 0
                for p in sorted(pred):
                    bj, bd = -1, tol + 1
                    for j, g in enumerate(gt):
                        if not matched[j] and abs(p - g) <= tol and abs(p - g) < bd:
                            bj, bd = j, abs(p - g)
                    if bj >= 0:
                        matched[bj] = True; tp += 1
                fp, fn = len(pred) - tp, len(gt) - tp
                return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
            for _dsname, _ds in [("in", inval_ds), ("ood", ood_ds)]:
                N = min(a.diag_rollout, len(_ds))
                for _T in [float(x) for x in str(a.diag_temps).split(",")]:
                    stds, means, uniq, npos = [], [], [], 0
                    for i in range(N):
                        ws = _ds[i]; gt = ws["target_frames"]
                        if not gt:                         # skip pure-negative (no GT) windows here
                            continue
                        npos += 1
                        rolls = loz.sample_rollouts(to_sample(ws), G=a.diag_G, temperature=_T)
                        rew = [_f1(fr, gt, 2) for fr, _ in rolls]
                        stds.append(float(_np.std(rew))); means.append(float(_np.mean(rew)))
                        uniq.append(len(set(tuple(sorted(fr)) for fr, _ in rolls)))
                    if npos:
                        frac = _np.mean([s > 1e-6 for s in stds])
                        log0(f"[diag:{_dsname}] T={_T} G={a.diag_G} N={npos} | reward_std_mean={_np.mean(stds):.3f} "
                             f"frac(std>0)={frac:.2f} | reward_mean={_np.mean(means):.3f} "
                             f"uniq_preds_mean={_np.mean(uniq):.2f}/{a.diag_G}")
            if is_main and wb is not None:
                wb.finish()
            if ddp:
                dist.barrier(); dist.destroy_process_group()
            return
        log0(f"[eval_only] evaluating connector {a.init_fc} @ train-step {step0} (fsdp={a.fsdp}, world={world})")
        run_eval(step0)                                    # per-dataset mAP -> wandb at step0 (overlay on loss run)
        if is_main and wb is not None:
            wb.finish()
        if ddp:
            dist.barrier(); dist.destroy_process_group()
        return

    # ---- preemption-safe resume: connector + optimizer + LR sched + step, on the PVC ----
    # The frozen Qwen reloads from the base each start (not saved); only the tiny trainable
    # state is checkpointed. Kueue can preempt this shared-cluster 4-GPU job mid-run; on
    # restart we continue from the last eval instead of step 0. All ranks load the same file
    # (fc/opt stay identical across ranks via the grad all-reduce), rank0 writes it atomically.
    resume_path = os.path.join(a.out, f"{a.run_name}_resume.pt")
    gstep = 0
    if os.path.exists(resume_path):
        rk = torch.load(resume_path, map_location="cpu", weights_only=False)
        fc.load_state_dict(rk["fc"])
        opt.load_state_dict(rk["opt"])
        for st in opt.state.values():                     # move Adam moments onto this rank's GPU
            for kk, vv in st.items():
                if isinstance(vv, torch.Tensor):
                    st[kk] = vv.to(dev)
        sched.load_state_dict(rk["sched"])
        gstep = int(rk["gstep"]); best["v"] = float(rk["best"])
        if lora_params and rk.get("lora"):
            for p, s in zip(lora_params, rk["lora"]):
                p.data.copy_(s.to(p.device, p.dtype))
        log0(f"[resume] {resume_path} @ step {gstep} (best in-mAP@2={best['v']:.2f}) -> continue")
    else:
        log0("[eval @ step 0 (init / before mixing)]")
        run_eval(0)

    def save_resume(step):
        if not is_main:
            return
        tmp = resume_path + ".tmp"
        rk = {"fc": fc.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
              "gstep": int(step), "best": float(best["v"])}
        if lora_params:
            rk["lora"] = [p.detach().cpu() for p in lora_params]
        torch.save(rk, tmp); os.replace(tmp, resume_path)   # atomic: never a half-written resume

    order0 = list(range(len(train_ds)))
    last_ck = time.time()                                 # wall-clock of last resume save (time-based ckpt)
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=a.num_workers) if a.num_workers > 0 else None
    if is_main:
        log0(f"[data] frame prefetch: {a.num_workers} threads" if pool else "[data] synchronous frame load")
    for ep in range(a.epochs):
        fc.train()
        if getattr(train_ds, "temperature_epoch_order", None) is not None:  # temp-weighted WITHOUT replacement
            order = train_ds.temperature_epoch_order(ep)             # deterministic on all ranks -> clean shard; full coverage
        elif getattr(train_ds, "sample_weights", None) is not None:  # legacy iid temperature sampling
            g = torch.Generator().manual_seed(1000 + ep)             # identical draw on all ranks -> clean shard
            order = torch.multinomial(train_ds.sample_weights, len(train_ds),
                                      replacement=True, generator=g).tolist()
        else:
            order = order0[:]; random.Random(1000 + ep).shuffle(order)
        order = order[rank::world]                       # DDP shard
        rl, t0, micro, step_loss = 0.0, time.time(), 0, 0.0
        opt.zero_grad(set_to_none=True); stop = False
        if pool is not None:                             # prefetch frame decode in bg threads
            # in-flight depth = num_workers so ALL decode threads stay busy (64-core node, tons of RAM)
            batch_iter = _prefetch_batches(train_ds, order, a.batch_size, pool,
                                           ahead=max(2, a.num_workers))
        else:
            batch_iter = ([train_ds[order[b0 + j]] for j in range(a.batch_size)]
                          for b0 in range(0, (len(order) // a.batch_size) * a.batch_size, a.batch_size))
        for wss in batch_iter:
            batch = [to_sample(ws) for ws in wss]
            loss = loz.loss_batch(batch) / a.grad_accum   # scale for accumulation
            loss.backward()
            lv = float(loss.detach()) * a.grad_accum
            rl += lv; step_loss += lv; micro += 1
            if micro % a.grad_accum != 0:                 # keep accumulating
                continue
            if ddp:                                       # all-reduce the small trainable grads
                for p in trainable:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM); p.grad /= world
            gnorm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0)); opt.step(); sched.step()
            opt.zero_grad(set_to_none=True)
            gstep += 1
            cur = step_loss / a.grad_accum; step_loss = 0.0        # THIS opt-step's loss (not cumulative)
            eff = a.batch_size * world * a.grad_accum
            if is_main and wb is not None:
                wb.log({"train/loss": cur, "train/loss_avg": rl / micro, "train/lr": sched.get_last_lr()[0],
                        "train/grad_norm": gnorm, "train/epoch": ep}, step=gstep)
            if gstep % 5 == 0:
                log0(f"  ep{ep} step{gstep} loss {cur:.4f} (avg {rl/micro:.4f}) gnorm {gnorm:.2f} | "
                     f"{gstep*eff/(time.time()-t0):.2f} samp/s eff_batch={eff}")
            if a.eval_every > 0 and gstep % a.eval_every == 0:
                run_eval(gstep)
                save_resume(gstep); last_ck = time.time()  # checkpoint after every eval (preempt-safe)
                save_conn(gstep)                           # named connector ckpt (kept) for offline OOD-select
            elif (a.ckpt_every and gstep % a.ckpt_every == 0) or (a.ckpt_secs and time.time() - last_ck >= a.ckpt_secs):
                save_resume(gstep); last_ck = time.time()  # step- OR time-based: preemption loses <= ckpt_secs of work
            if a.max_steps and gstep >= a.max_steps:
                stop = True; break
        if stop:
            break
    run_eval(gstep)
    save_resume(gstep)
    if is_main:
        if wb is not None:
            wb.finish()
        log0(f"[done] {a.run_name} best in-domain mAP@2 {best['v']:.2f}")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
