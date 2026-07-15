"""Train the Action Head alone (backbone frozen). The head refines each coarse <idx>
prediction of the frozen generation model into a frame-level timestamp.

Backbone (Qwen, V-JEPA adaptor, LoRA) all frozen; only the ~3.5M-param head updates.

TRAIN DATA = PRED-CENTRIC pairs (修1): for each real train PREDICTION p whose nearest
same-type GT g is within ±W, one training example (center=p, target=g-p). This is
EXACTLY the distribution the head sees at inference (it refines predictions). Far/FP
predictions (|g-p|>W) are skipped — the head is not expected to fix them (no GT, no 0).

EVAL (修2): both baseline and refined detections pass the SAME score-based 1D NMS
(same-label ±win keep-highest) so refined-collapse doesn't inflate/deflate mAP.

query (修3/修4): last-token LLM hidden of GENERIC_Q[type] (context-free, cacheable) —
train AND eval use GENERIC_Q to remove the paraphrase variable while validating the head.
"""
from __future__ import annotations

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
sys.path.insert(0, _COMMON)
FEAT = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/feat_interleave"

from data.questions import GENERIC_Q                           # noqa: E402
from common.score import compute_mAPs                          # noqa: E402


def _feats_cache():
    c = {}

    def get(vid):
        if vid not in c:
            c[vid] = np.load(os.path.join(FEAT, vid + ".npy")).astype(np.float32)
        return c[vid]
    return get


def nms_1d(events, win=2):
    """Same-label score NMS: within ±win frames keep only the highest-score event."""
    out = []
    for lab in set(e["label"] for e in events):
        kept = []
        for e in sorted([e for e in events if e["label"] == lab], key=lambda x: -x["score"]):
            if all(abs(e["frame"] - k["frame"]) > win for k in kept):
                kept.append(e)
        out += kept
    return out


def score_mAP(dets, truth, tol=(0, 1, 2, 4)):
    order = sorted(truth)
    tl = [{"video": v, "events": truth[v]} for v in order]
    pl = [{"video": v, "events": dets.get(v, [])} for v in order]
    mAPs, _ = compute_mAPs(tl, pl, tolerances=list(tol))
    return {t: float(m) * 100 for t, m in zip(tol, mAPs)}


def run(gen_ckpt=os.path.join(ROOT, "outputs/idx/idx_multi/best.pt"),
        W_win=24, d=256, n_heads=4, epochs=20, batch_size=128, lr=3e-4,
        pool_videos=800, nms_win=2, query_mode="cfree", run_name="action_head",
        huber_beta=1.0, seed=0):
    from models.wrapper import QwenWrapper
    from models.vjepa_adaptor import VJEPAAdaptor
    from models.idx_localizer import IdxLocalizer
    from models.action_head import ActionHead

    torch.manual_seed(seed)
    dev = "cuda"
    fps = 15
    feats_of = _feats_cache()
    types = ("touch", "untouch")

    # ---- frozen backbone ----
    ck = torch.load(gen_ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]; rank = cfg.get("lora_rank", 16)
    W = QwenWrapper(device=dev, dtype=torch.bfloat16)
    ad = VJEPAAdaptor(768, W.d_llm, hidden=2048, n_layers=2).to(dev, torch.bfloat16)
    ad.set_target_rms_from(W.embed_tokens.weight)             # NOT in state_dict!
    ad.load_state_dict(ck["adaptor"]); ad.eval()
    for p in ad.parameters():
        p.requires_grad_(False)
    if cfg.get("lora"):
        lp = W.add_lora(rank=rank, alpha=2 * rank, n_layers=W.n_layers, target="all")
        for p, s in zip(lp, ck["lora"]):
            p.data.copy_(s.to(p.device, p.dtype))
    loz = IdxLocalizer(W, ad, use_idx=True, use_anchor=True, anchor_stride=5,
                       anchor_max_side=252, use_mrope=False, fps=fps, max_frames=320,
                       grad_checkpoint=False)

    head = ActionHead(W.d_llm, d=d, n_heads=n_heads, window=W_win).to(dev)
    print(f"[cfg] action-head d={d} heads={n_heads} W={W_win} "
          f"params={sum(p.numel() for p in head.parameters())/1e6:.2f}M "
          f"bs={batch_size} lr={lr} nms_win={nms_win}", flush=True)

    # ---- language query cache: last-token LLM hidden (context-free; 修4) ----
    qcache = {}

    @torch.no_grad()
    def qembed(question):
        if question not in qcache:
            ids = W.tokenizer(question, return_tensors="pt",
                              add_special_tokens=False).input_ids.to(dev)
            out = W.llm(inputs_embeds=W.embed_tokens(ids), use_cache=False, return_dict=True)
            qcache[question] = out.last_hidden_state[0, -1].float()
        return qcache[question]

    # ---- window builder: pre-LLM adaptor V over the ±W frames (pure vision, no idx) ----
    L = 2 * W_win + 1

    def build_V(centers, feats_list):
        B = len(centers)
        win = torch.zeros(B, L, 768, dtype=torch.float32)
        mask = torch.ones(B, L, dtype=torch.bool)             # True = pad
        for i, (c, f) in enumerate(zip(centers, feats_list)):
            N = f.shape[0]
            for k in range(L):
                fr = c - W_win + k
                if 0 <= fr < N:
                    win[i, k] = torch.from_numpy(f[fr]); mask[i, k] = False
        return ad(win.to(dev, torch.bfloat16)).float(), mask.to(dev)

    # ---- generation on a split (cached to disk) ----
    @torch.no_grad()
    def generate_split(split, max_videos=0):
        vids = json.load(open(os.path.join(ROOT, "data", "annotations", f"{split}.json")))
        vids = [v for v in vids if os.path.exists(os.path.join(FEAT, v["video_id"] + ".npy"))]
        if max_videos:
            vids = vids[:max_videos]
        items = [(v["video_id"], t) for v in vids for t in types
                 if any(e["type"] == t for e in v["events"])]
        out = {}
        for b0 in range(0, len(items), 8):
            chunk = items[b0:b0 + 8]
            samples = [{"feats": torch.from_numpy(feats_of(vid)), "question": GENERIC_Q[t],
                        "event_frames": None, "gt": -1, "fps": fps, "type": t, "video_id": vid,
                        "num_frames": feats_of(vid).shape[0], "full_num_frames": feats_of(vid).shape[0],
                        "anchor_start_sec": 0, "anchor_num_secs": feats_of(vid).shape[0] // fps}
                       for vid, t in chunk]
            for (vid, t), (fr, sc) in zip(chunk, loz.predict_multievent_batch(samples)):
                N = feats_of(vid).shape[0]
                out[(vid, t)] = [(int(f), float(s)) for f, s in zip(fr, sc) if 0 <= f < N]
        return out, vids

    cache_dir = os.path.join(ROOT, "outputs", "action", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    gtag = os.path.basename(os.path.dirname(gen_ckpt))

    def cached_generate(split, max_videos=0):
        path = os.path.join(cache_dir, f"{gtag}_{split}_{max_videos}.pt")
        if os.path.exists(path):
            d = torch.load(path, weights_only=False)
            print(f"[cache] loaded {os.path.basename(path)}", flush=True)
            return d["coarse"], d["vids"]
        print(f"[gen] generating {split} (max_videos={max_videos})...", flush=True)
        coarse, vids = generate_split(split, max_videos)
        torch.save({"coarse": coarse, "vids": vids}, path)
        return coarse, vids

    # ---- val: baseline dets (NMS'd; 修2) ----
    coarse_val, val_vids = cached_generate("val")
    truth_val = {v["video_id"]: [{"label": e["type"], "frame": e["frame"]}
                                 for e in v["events"]] for v in val_vids}
    base_raw = {v: [] for v in truth_val}
    for (vid, t), lst in coarse_val.items():
        for idx, sc in lst:
            base_raw[vid].append({"label": t, "frame": idx, "score": sc})
    base_dets = {v: nms_1d(evs, nms_win) for v, evs in base_raw.items()}
    base_ap = score_mAP(base_dets, truth_val)
    print(f"[baseline gen-only, NMS] mAP@0/1/2/4 = {[round(base_ap[t],2) for t in (0,1,2,4)]}", flush=True)

    # ---- train: PRED-CENTRIC pairs (修1) ----
    coarse_tr, tr_vids = cached_generate("train", max_videos=pool_videos)
    pairs = []                                    # (vid, type, center=pred, target=gt-pred)
    for v in tr_vids:
        for t in types:
            gts = sorted(e["frame"] for e in v["events"] if e["type"] == t)
            for p, _ in coarse_tr.get((v["video_id"], t), []):
                if not gts:
                    continue
                off = min(gts, key=lambda x: abs(x - p)) - p     # gt - pred (signed)
                if abs(off) <= W_win:                            # rescuable -> train; else skip
                    pairs.append((v["video_id"], t, int(p), int(off)))
    offs = np.array([o for _, _, _, o in pairs])
    print(f"[pairs] n={len(pairs)} | within2={100*(np.abs(offs)<=2).mean():.0f}% "
          f"within4={100*(np.abs(offs)<=4).mean():.0f}% within8={100*(np.abs(offs)<=8).mean():.0f}% "
          f"std={offs.std():.1f} | (@0 ceiling ~ within-W fraction)", flush=True)

    # ---- refine val + NMS -> mAP ----
    @torch.no_grad()
    def refine_and_score():
        head.eval()
        flat = [(vid, t, idx, sc) for (vid, t), lst in coarse_val.items() for idx, sc in lst]
        dets = {v: [] for v in truth_val}
        for b0 in range(0, len(flat), 256):
            chunk = flat[b0:b0 + 256]
            V, mask = build_V([c for _, _, c, _ in chunk], [feats_of(v) for v, _, _, _ in chunk])
            q = torch.stack([qembed(GENERIC_Q[t]) for _, t, _, _ in chunk])
            delta = head(q, V, mask).cpu().numpy()
            for (vid, t, idx, sc), dl in zip(chunk, delta):
                N = feats_of(vid).shape[0]
                dets[vid].append({"label": t, "frame": int(round(min(max(idx + dl, 0), N - 1))),
                                  "score": sc})
        dets = {v: nms_1d(evs, nms_win) for v, evs in dets.items()}
        return score_mAP(dets, truth_val)

    # ---- train loop ----
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    steps = (len(pairs) // batch_size) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    hub = torch.nn.SmoothL1Loss(beta=huber_beta)
    out_dir = os.path.join(ROOT, "outputs", "action", run_name)
    os.makedirs(out_dir, exist_ok=True)
    mpath = os.path.join(out_dir, "metrics.csv")
    open(mpath, "w").write("epoch,loss,mAP@0,mAP@1,mAP@2,mAP@4,base@0,base@2,base@4\n")
    best = -1.0
    for ep in range(epochs):
        head.train()
        prng = random.Random(1000 + ep)
        order = pairs[:]; prng.shuffle(order)
        run_loss, t0, nb = 0.0, time.time(), 0
        for b0 in range(0, (len(order) // batch_size) * batch_size, batch_size):
            chunk = order[b0:b0 + batch_size]
            V, mask = build_V([c for _, _, c, _ in chunk], [feats_of(v) for v, _, _, _ in chunk])
            q = torch.stack([qembed(GENERIC_Q[t]) for _, t, _, _ in chunk])
            delta = head(q, V, mask)
            tgt = torch.tensor([o for _, _, _, o in chunk], device=dev, dtype=torch.float32)
            loss = hub(delta, tgt)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step(); sched.step()
            run_loss += float(loss.detach()); nb += 1
        ap = refine_and_score()
        print(f"[ep{ep}] loss {run_loss/max(nb,1):.3f} | REFINED mAP@0/1/2/4 = "
              f"{[round(ap[t],2) for t in (0,1,2,4)]} | base "
              f"{base_ap[0]:.1f}/{base_ap[2]:.1f}/{base_ap[4]:.1f} "
              f"| {len(order)/(time.time()-t0):.0f} pairs/s", flush=True)
        open(mpath, "a").write(f"{ep},{run_loss/max(nb,1):.3f},{ap[0]:.2f},{ap[1]:.2f},"
                               f"{ap[2]:.2f},{ap[4]:.2f},{base_ap[0]:.2f},{base_ap[2]:.2f},{base_ap[4]:.2f}\n")
        if ap[2] > best:
            best = ap[2]
            torch.save({"head": head.state_dict(), "cfg": dict(d=d, n_heads=n_heads,
                        window=W_win, gen_ckpt=gen_ckpt), "ap": ap, "base_ap": base_ap},
                       os.path.join(out_dir, "best.pt"))
    print(f"[done] best REFINED mAP@2 {best:.2f} (baseline {base_ap[2]:.2f})", flush=True)
    return best


if __name__ == "__main__":
    run()
