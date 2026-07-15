"""Stage-2 refiner training (backbone frozen; only the refiner updates).

Stage 1 = frozen idx-generation (coarse <idx>, counting, discrimination — 44.9).
Stage 2 = language-conditioned SALIENCY refiner over the ±W window: per-frame
saliency -> soft-argmax -> refined frame = idx + δ.

query = GROUNDED language hidden: the question-position (prefix last-token) hidden
from the FULL-sequence forward (it has attended to THIS video's motion), precomputed
once and cached. Train data = PRED-CENTRIC pairs (center=pred, target=gt-pred, |.|<=W).
Loss = soft-CE(saliency, Gaussian@target) + λ·L1(soft-argmax δ, target). Eval: refine
val dets + same 1D-NMS as baseline -> mAP@{0,1,2,4}.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/home/chang_noroot/data2/hf_cache")
_COMMON = "/home/chang_noroot/data2/huyanh/Workspace/hand_touch_detection"
sys.path.insert(0, _COMMON)
FEAT = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/feat_interleave"

from data.questions import GENERIC_Q                          # noqa: E402
from common.score import compute_mAPs                         # noqa: E402
from train.action_train import _feats_cache, nms_1d, score_mAP   # noqa: E402
from models.action_refine import build_refiner, soft_argmax     # noqa: E402


def run(gen_ckpt=os.path.join(ROOT, "outputs/idx/idx_multi/best.pt"),
        head_type="tcn", W_win=12, d=256, n_layers=6, epochs=25, batch_size=128,
        lr=5e-4, pool_videos=800, nms_win=2, sigma=1.5, lam_l1=0.5, query_mode="grounded",
        run_name="refine", seed=0):
    from models.wrapper import QwenWrapper
    from models.vjepa_adaptor import VJEPAAdaptor
    from models.idx_localizer import IdxLocalizer

    torch.manual_seed(seed)
    dev = "cuda"; fps = 15
    feats_of = _feats_cache()
    types = ("touch", "untouch")

    ck = torch.load(gen_ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]; rank = cfg.get("lora_rank", 16)
    W = QwenWrapper(device=dev, dtype=torch.bfloat16)
    ad = VJEPAAdaptor(768, W.d_llm, hidden=2048, n_layers=2).to(dev, torch.bfloat16)
    ad.set_target_rms_from(W.embed_tokens.weight); ad.load_state_dict(ck["adaptor"]); ad.eval()
    for p in ad.parameters():
        p.requires_grad_(False)
    if cfg.get("lora"):
        lp = W.add_lora(rank=rank, alpha=2 * rank, n_layers=W.n_layers, target="all")
        for p, s in zip(lp, ck["lora"]):
            p.data.copy_(s.to(p.device, p.dtype))
    loz = IdxLocalizer(W, ad, use_idx=True, use_anchor=True, anchor_stride=5,
                       anchor_max_side=252, use_mrope=False, fps=fps, max_frames=320,
                       grad_checkpoint=False)

    refiner = build_refiner(head_type, W.d_llm, d, W_win, n_layers).to(dev)
    print(f"[cfg] refine head={head_type} W={W_win} d={d} L={n_layers} "
          f"params={sum(p.numel() for p in refiner.parameters())/1e6:.2f}M query={query_mode} "
          f"lr={lr} sigma={sigma} lam_l1={lam_l1}", flush=True)

    L = 2 * W_win + 1
    cache_dir = os.path.join(ROOT, "outputs", "action", "cache"); os.makedirs(cache_dir, exist_ok=True)
    gtag = os.path.basename(os.path.dirname(gen_ckpt))

    def build_V(centers, feats_list):
        B = len(centers)
        win = torch.zeros(B, L, 768); mask = torch.ones(B, L, dtype=torch.bool)
        for i, (c, f) in enumerate(zip(centers, feats_list)):
            N = f.shape[0]
            for k in range(L):
                fr = c - W_win + k
                if 0 <= fr < N:
                    win[i, k] = torch.from_numpy(f[fr]); mask[i, k] = False
        return ad(win.to(dev, torch.bfloat16)).float(), mask.to(dev)

    # ---- generation (cached) ----
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

    def cached_generate(split, max_videos=0):
        path = os.path.join(cache_dir, f"{gtag}_{split}_{max_videos}.pt")
        if os.path.exists(path):
            dd = torch.load(path, weights_only=False)
            print(f"[cache] {os.path.basename(path)}", flush=True); return dd["coarse"], dd["vids"]
        print(f"[gen] {split} ({max_videos})...", flush=True)
        coarse, vids = generate_split(split, max_videos)
        torch.save({"coarse": coarse, "vids": vids}, path); return coarse, vids

    # ---- GROUNDED query: prefix last-token hidden (attended to this video), cached ----
    gq_path = os.path.join(cache_dir, f"{gtag}_groundedq.pt")
    gq = torch.load(gq_path, weights_only=False) if (query_mode == "grounded" and os.path.exists(gq_path)) else {}
    qcache = {}

    @torch.no_grad()
    def cfree_q(typ):
        if typ not in qcache:
            ids = W.tokenizer(GENERIC_Q[typ], return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
            qcache[typ] = W.llm(inputs_embeds=W.embed_tokens(ids), use_cache=False,
                                return_dict=True).last_hidden_state[0, -1].float()
        return qcache[typ]

    @torch.no_grad()
    def precompute_grounded(need_items):
        todo = [it for it in need_items if f"{it[0]}|{it[1]}" not in gq]
        if not todo:
            return
        print(f"[grounded] computing {len(todo)} prefix hiddens...", flush=True)
        for b0 in range(0, len(todo), 8):
            chunk = todo[b0:b0 + 8]
            embs = []
            for vid, t in chunk:
                f = feats_of(vid); N = f.shape[0]
                s = {"feats": torch.from_numpy(f), "question": GENERIC_Q[t], "event_frames": None,
                     "gt": -1, "fps": fps, "type": t, "video_id": vid, "num_frames": N,
                     "full_num_frames": N, "anchor_start_sec": 0, "anchor_num_secs": N // fps}
                embs.append(loz._build(s, with_answer=False)[0])
            S = max(e.shape[0] for e in embs); B = len(embs); dd = embs[0].shape[-1]
            emb = torch.zeros(B, S, dd, device=dev, dtype=embs[0].dtype)
            att = torch.zeros(B, S, device=dev, dtype=torch.long)
            for i, e in enumerate(embs):
                emb[i, :e.shape[0]] = e; att[i, :e.shape[0]] = 1
            out = W.llm(inputs_embeds=emb, attention_mask=att, use_cache=False, return_dict=True)
            for i, (vid, t) in enumerate(chunk):
                gq[f"{vid}|{t}"] = out.last_hidden_state[i, embs[i].shape[0] - 1].float().cpu()
        torch.save(gq, gq_path)

    def get_q(vid, typ):
        if query_mode == "grounded":
            return gq[f"{vid}|{typ}"].to(dev)
        return cfree_q(typ)

    # ---- data ----
    coarse_val, val_vids = cached_generate("val")
    truth_val = {v["video_id"]: [{"label": e["type"], "frame": e["frame"]} for e in v["events"]]
                 for v in val_vids}
    base_raw = {v: [] for v in truth_val}
    for (vid, t), lst in coarse_val.items():
        for idx, sc in lst:
            base_raw[vid].append({"label": t, "frame": idx, "score": sc})
    base_ap = score_mAP({v: nms_1d(e, nms_win) for v, e in base_raw.items()}, truth_val)
    print(f"[baseline NMS] mAP@0/1/2/4 = {[round(base_ap[t],2) for t in (0,1,2,4)]}", flush=True)

    coarse_tr, tr_vids = cached_generate("train", max_videos=pool_videos)
    pairs = []
    for v in tr_vids:
        for t in types:
            gts = sorted(e["frame"] for e in v["events"] if e["type"] == t)
            for p, _ in coarse_tr.get((v["video_id"], t), []):
                if gts:
                    off = min(gts, key=lambda x: abs(x - p)) - p
                    if abs(off) <= W_win:
                        pairs.append((v["video_id"], t, int(p), int(off)))
    print(f"[pairs] n={len(pairs)}", flush=True)
    if query_mode == "grounded":
        need = list({(vid, t) for vid, t, _, _ in pairs} |
                    {(vid, t) for (vid, t) in coarse_val.keys()})
        precompute_grounded(need)

    pos = torch.arange(-W_win, W_win + 1, device=dev, dtype=torch.float32)

    def sal_target(offsets):                                 # [B] -> soft Gaussian [B,L]
        o = torch.tensor(offsets, device=dev, dtype=torch.float32).unsqueeze(1)   # [B,1]
        t = torch.exp(-((pos.unsqueeze(0) - o) ** 2) / (2 * sigma ** 2))          # [B,L]
        return t / (t.sum(-1, keepdim=True) + 1e-8)

    @torch.no_grad()
    def refine_and_score():
        refiner.eval()
        flat = [(vid, t, idx, sc) for (vid, t), lst in coarse_val.items() for idx, sc in lst]
        dets = {v: [] for v in truth_val}
        for b0 in range(0, len(flat), 256):
            chunk = flat[b0:b0 + 256]
            V, mask = build_V([c for _, _, c, _ in chunk], [feats_of(v) for v, _, _, _ in chunk])
            q = torch.stack([get_q(v, t) for v, t, _, _ in chunk])
            sal = refiner(q, V, mask)
            delta, _ = soft_argmax(sal, mask, W_win)
            delta = delta.cpu().numpy()
            for (vid, t, idx, sc), dl in zip(chunk, delta):
                N = feats_of(vid).shape[0]
                dets[vid].append({"label": t, "frame": int(round(min(max(idx + dl, 0), N - 1))), "score": sc})
        return score_mAP({v: nms_1d(e, nms_win) for v, e in dets.items()}, truth_val)

    opt = torch.optim.AdamW(refiner.parameters(), lr=lr, weight_decay=0.01)
    steps = (len(pairs) // batch_size) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    out_dir = os.path.join(ROOT, "outputs", "action", run_name); os.makedirs(out_dir, exist_ok=True)
    mpath = os.path.join(out_dir, "metrics.csv")
    open(mpath, "w").write("epoch,loss,mAP@0,mAP@1,mAP@2,mAP@4,base@2,base@4\n")
    best = -1.0
    for ep in range(epochs):
        refiner.train()
        prng = random.Random(1000 + ep); order = pairs[:]; prng.shuffle(order)
        run_loss, t0, nb = 0.0, time.time(), 0
        for b0 in range(0, (len(order) // batch_size) * batch_size, batch_size):
            chunk = order[b0:b0 + batch_size]
            V, mask = build_V([c for _, _, c, _ in chunk], [feats_of(v) for v, _, _, _ in chunk])
            q = torch.stack([get_q(v, t) for v, t, _, _ in chunk])
            offs = [o for _, _, _, o in chunk]
            sal = refiner(q, V, mask)                        # [B,L] logits
            tgt = sal_target(offs)                           # [B,L] soft
            logp = F.log_softmax(sal, dim=-1)
            ce = -(tgt * logp).sum(-1).mean()                # soft-CE
            delta, _ = soft_argmax(sal, mask, W_win)
            l1 = F.smooth_l1_loss(delta, torch.tensor(offs, device=dev, dtype=torch.float32))
            loss = ce + lam_l1 * l1
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
            opt.step(); sched.step()
            run_loss += float(loss.detach()); nb += 1
        ap = refine_and_score()
        print(f"[ep{ep}] loss {run_loss/max(nb,1):.3f} | REFINED @0/1/2/4 = "
              f"{[round(ap[t],2) for t in (0,1,2,4)]} | base @2/@4 {base_ap[2]:.1f}/{base_ap[4]:.1f} "
              f"| {len(order)/(time.time()-t0):.0f} pairs/s", flush=True)
        open(mpath, "a").write(f"{ep},{run_loss/max(nb,1):.3f},{ap[0]:.2f},{ap[1]:.2f},{ap[2]:.2f},"
                               f"{ap[4]:.2f},{base_ap[2]:.2f},{base_ap[4]:.2f}\n")
        if ap[2] > best:
            best = ap[2]
            torch.save({"refiner": refiner.state_dict(), "head_type": head_type,
                        "cfg": dict(d=d, n_layers=n_layers, window=W_win), "ap": ap, "base_ap": base_ap},
                       os.path.join(out_dir, "best.pt"))
    print(f"[done] {run_name} best REFINED mAP@2 {best:.2f} (baseline {base_ap[2]:.2f}) "
          f"@0 base {base_ap[0]:.2f}", flush=True)
    return best


if __name__ == "__main__":
    run()
