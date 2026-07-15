"""RLVR (GRPO) post-training for Stage-1 idx generation, on top of M1 (query-compression
connector, frozen-LLM 67.6). Fixes the SFT metric-blindness: reward = how CLOSE the
generated idx are to GT (verifiable, no reward model). GRPO: K samples/prompt, group-
normalized advantage, policy-gradient on LoRA(+connector).

Reward schemes (--reward): tiered (mAP-tolerance tiers) | smooth (exp distance) | map.
Init: --init_fc M1 compressor  [+ optional --init_lora from an SFT-LoRA ckpt for #3].
OOM-safe: 1 prompt/step, K rollouts, log-prob forward chunked, grad-checkpoint LLM.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import re
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/home/chang_noroot/data2/hf_cache")
_COMMON = "/home/chang_noroot/data2/huyanh/Workspace/hand_touch_detection"
sys.path.insert(0, _COMMON); sys.path.insert(0, os.path.join(_COMMON, "methods/spot_head"))
LAB = os.path.join(_COMMON, "data", "HOI4D-v3")
GRID = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/grid_even"
import eval_nms as en                                             # noqa: E402
from data.questions import GENERIC_Q                              # noqa: E402


def match_reward(pred_frames, gt_frames, scheme, sigma=2.0):
    """Greedy-match preds to GTs by nearest; reward shaped per --scheme + miss/extra penalties.
    dists = matched-pred distances; n_extra = preds with no GT left; n_miss = unmatched GTs."""
    gts = sorted(gt_frames); used = [False] * len(gts); dists = []
    for p in pred_frames:
        best, bj = 1e9, -1
        for j, g in enumerate(gts):
            if not used[j] and abs(p - g) < best:
                best, bj = abs(p - g), j
        if bj < 0:
            dists.append(None); continue                         # extra pred, no GT left
        used[bj] = True; dists.append(best)
    n_extra = sum(d is None for d in dists); n_miss = used.count(False)
    matched = [d for d in dists if d is not None]

    if scheme == "f1":                                           # count/recall-balanced (enumeration)
        tp = sum(d <= 1 for d in matched)                        # true-pos within tol 1
        ndet, ngt = len(pred_frames), len(gts)
        P = tp / ndet if ndet else 0.0; R = tp / ngt if ngt else 0.0
        f1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
        exact = float(np.mean([np.exp(-d) for d in matched])) if matched else 0.0
        return f1 + 0.3 * exact                                  # F1 + tight-exactness bonus
    if scheme == "sharp":                                        # exactness -> score calibration (@0/@1)
        r = sum(1.0 if d == 0 else 0.35 if d <= 1 else 0.1 if d <= 2 else 0.0 for d in matched)
        return r - 0.5 * n_extra - 0.3 * n_miss                  # heavy extra penalty (scored FPs hurt mAP)
    if scheme == "graded":                                       # @2-aligned: credit within-2, keep recall
        r = sum(1.5 if d == 0 else 1.2 if d <= 1 else 1.0 if d <= 2 else 0.0 for d in matched)
        return r - 0.1 * n_extra - 0.3 * n_miss                  # LIGHT extra penalty (low-conf FPs rank low)

    r = 0.0                                                       # tiered | smooth | map (distance-shaped)
    for e in matched:
        if scheme == "tiered":
            r += 1.0 if e == 0 else 0.7 if e <= 1 else 0.5 if e <= 2 else 0.2 if e <= 4 else 0.0
        elif scheme == "smooth":
            r += float(np.exp(-e / sigma))
        else:                                                    # "map": credit if within tol 2
            r += 1.0 if e <= 2 else 0.0
    return r - 0.2 * n_extra - 0.3 * n_miss                      # extra + missed-GT (recall) penalties


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--init_fc", default="outputs/idx_compress/comp_notext/best.pt")
    ap.add_argument("--init_lora", default="")                   # for #3: SFT-LoRA -> RLVR
    ap.add_argument("--reward", default="tiered", choices=["tiered", "smooth", "map", "sharp", "f1", "graded"])
    ap.add_argument("--K", type=int, default=6); ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--batch_prompts", type=int, default=4); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lp_chunk", type=int, default=2); ap.add_argument("--kl", type=float, default=0.0)
    ap.add_argument("--evals_per_epoch", type=int, default=2); ap.add_argument("--eval_videos", type=int, default=138)
    ap.add_argument("--cpu_threads", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_limit", type=int, default=0); ap.add_argument("--max_new", type=int, default=24)
    a = ap.parse_args()
    torch.manual_seed(a.seed); torch.set_num_threads(a.cpu_threads); dev = "cuda"
    en.TOLS = [0, 1, 2]

    from models.wrapper import QwenWrapper
    from models.vjepa_adaptor import VJEPAAdaptor
    from models.idx_localizer import IdxLocalizer
    from models.frame_compress import FrameCompress

    W = QwenWrapper(device=dev, dtype=torch.bfloat16)
    ad = VJEPAAdaptor(768, W.d_llm, hidden=2048, n_layers=2).to(dev, torch.bfloat16)
    fc = FrameCompress(768, 8, W.d_llm, n_heads=4, stain=False, use_text=False).to(dev, torch.float32)
    fc.set_target_rms_from(W.embed_tokens.weight)
    fc.load_state_dict(torch.load(a.init_fc, weights_only=False)["fc"])   # M1 connector
    lora_params = W.add_lora(rank=16, alpha=32, n_layers=W.n_layers, target="all")
    if a.init_lora:                                                       # #3: continue from SFT-LoRA
        sd = torch.load(a.init_lora, weights_only=False)
        for p, s in zip(lora_params, sd["lora"]):
            p.data.copy_(s.to(p.device, p.dtype))
        if "fc" in sd:
            fc.load_state_dict(sd["fc"])
    # KL reference = the SFT-init policy (frozen snapshot of LoRA + fc). Anchoring to it
    # stops policy-gradient from drifting off SFT's greedy @2 optimum while reward nudges.
    ref_lora = [p.detach().clone() for p in lora_params] if a.kl > 0 else None
    ref_fc = {k: v.detach().clone() for k, v in fc.state_dict().items()} if a.kl > 0 else None
    loz = IdxLocalizer(W, ad, use_idx=True, use_anchor=True, anchor_stride=5, anchor_max_side=252,
                       use_mrope=False, fps=15, max_frames=320, grad_checkpoint=True, compress=fc)
    trainable = list(fc.parameters()) + lora_params
    print(f"[cfg] RLVR reward={a.reward} K={a.K} temp={a.temp} lr={a.lr} | init_fc={os.path.basename(a.init_fc)} "
          f"init_lora={'yes' if a.init_lora else 'no'} | trainable={sum(p.numel() for p in trainable)/1e6:.1f}M", flush=True)

    def gload(k):
        g = np.load(os.path.join(GRID, k + ".npy")); return g, min(300, 2 * g.shape[0])

    tr = json.load(open(os.path.join(LAB, "train.json")))
    tr_items = [(v["video"].replace("/", "__"), t, sorted(e["frame"] for e in v["events"] if e["label"] == t))
                for v in tr if os.path.exists(os.path.join(GRID, v["video"].replace("/", "__") + ".npy"))
                for t in ("touch", "untouch") if any(e["label"] == t for e in v["events"])]
    val_vids = [v for v in json.load(open(os.path.join(LAB, "val.json")))
                if os.path.exists(os.path.join(GRID, v["video"].replace("/", "__") + ".npy"))][:a.eval_videos]
    truth = [{"video": v["video"], "num_frames": v["num_frames"],
              "events": [{"label": e["label"], "frame": e["frame"]} for e in v["events"]]} for v in val_vids]

    def sample_of(k, t):
        g, N = gload(k)
        return {"grid": torch.from_numpy(g), "question": GENERIC_Q[t], "type": t, "video_id": k,
                "fps": 15, "num_frames": N, "full_num_frames": N, "anchor_start_sec": 0,
                "anchor_num_secs": N // 15, "event_frames": None, "gt": -1}

    @torch.no_grad()
    def evaluate():
        loz.W.model.eval(); fc.eval()
        pr = {v["video"]: [] for v in val_vids}
        items = [(v, t) for v in val_vids for t in ("touch", "untouch") if any(e["label"] == t for e in v["events"])]
        for b0 in range(0, len(items), 2):
            ch = items[b0:b0 + 2]
            samps = [sample_of(v["video"].replace("/", "__"), t) for v, t in ch]
            for (v, t), (fr, sc) in zip(ch, loz.predict_multievent_batch(samps)):
                for f2, s2 in zip(fr, sc):
                    if 0 <= f2 < v["num_frames"]:
                        pr[v["video"]].append({"label": t, "frame": int(f2), "score": float(s2)})
        return en.maps_quiet(truth, [{"video": v["video"], "events": pr[v["video"]]} for v in val_vids])

    @torch.no_grad()
    def rollout(samples, K):
        """K sampled completions for EACH of B prompts -> list length B, each a list of K (frames, ids).
        Generation looped per-prompt (robust with inputs_embeds); training/update stays batched."""
        loz.W.model.eval(); fc.eval()
        groups = []
        for s in samples:
            emb, att, pos = loz._left_pad([s])                   # single prompt
            gen = loz.W.model.generate(inputs_embeds=emb, attention_mask=att,
                                       max_new_tokens=a.max_new, do_sample=True, temperature=a.temp, top_p=0.95,
                                       num_return_sequences=K, eos_token_id=loz.eos_id, pad_token_id=loz.eos_id,
                                       return_dict_in_generate=True)              # (K, T)
            g = []
            for j in range(K):
                ids = gen.sequences[j].tolist()
                if loz.eos_id in ids:
                    ids = ids[:ids.index(loz.eos_id)]
                frames = [int(x) for x in re.findall(r"\d+", loz.W.tokenizer.decode(ids))]
                g.append((frames, ids))
            groups.append(g)
        return groups

    def logprob_batch(pairs):
        """BATCHED length-normalized log-prob of completions (WITH grad). pairs: [(sample, comp_ids)]."""
        built = [(loz._build(s, with_answer=False)[0], loz.embed(torch.tensor(cids, device=dev)), cids)
                 for s, cids in pairs]
        Pn = [pe.shape[0] for pe, _, _ in built]; Cn = [len(c) for _, _, c in built]
        Ln = [p + c for p, c in zip(Pn, Cn)]; maxL = max(Ln); d = built[0][0].shape[-1]; nb = len(built)
        emb = torch.zeros(nb, maxL, d, device=dev, dtype=built[0][0].dtype)
        att = torch.zeros(nb, maxL, device=dev, dtype=torch.long)
        for i, (pe, ce, _) in enumerate(built):
            emb[i, :Ln[i]] = torch.cat([pe, ce], 0); att[i, :Ln[i]] = 1
        logits = loz.W.model(inputs_embeds=emb, attention_mask=att, use_cache=False).logits  # (nb,maxL,V)
        lps = []
        for i, (_, _, cids) in enumerate(built):
            pos = torch.arange(Pn[i] - 1, Pn[i] - 1 + Cn[i], device=dev)
            lg = F.log_softmax(logits[i, pos].float(), -1)                                   # (Cn, V)
            lp = lg[torch.arange(Cn[i], device=dev), torch.tensor(cids, device=dev)].sum()
            lps.append(lp / max(Cn[i], 1))
        return lps

    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.0)
    out_dir = os.path.join(ROOT, "outputs", "idx_rlvr", a.run_name); os.makedirs(out_dir, exist_ok=True)
    mp = os.path.join(out_dir, "metrics.csv"); open(mp, "w").write("step,reward,mAP0,mAP1,mAP2\n")
    def do_eval(tag, gstep, rmean):
        m = evaluate()
        print(f"[{tag}] reward~{rmean:.3f} | mAP@0/1/2={[round(x,2) for x in m]} "
              f"| {gstep/(time.time()-t0):.2f} step/s", flush=True)
        open(mp, "a").write(f"{gstep},{rmean:.3f},{m[0]:.2f},{m[1]:.2f},{m[2]:.2f}\n")
        nonlocal best
        if m[2] > best:
            best = m[2]
            torch.save({"fc": fc.state_dict(), "lora": [p.detach().cpu() for p in lora_params], "m": m},
                       os.path.join(out_dir, "best.pt"))

    m = evaluate(); print(f"[init] mAP@0/1/2={[round(x,2) for x in m]}", flush=True)
    best = m[2]; B = a.batch_prompts
    items = [it for it in tr_items if it[2]]                      # drop empty-GT
    if a.train_limit:
        items = items[:a.train_limit]
    nstep = len(items) // B
    eval_at = {int((e + 1) * nstep / a.evals_per_epoch) - 1 for e in range(a.evals_per_epoch)}
    rmean, t0, gstep = 0.0, time.time(), 0
    for ep in range(a.epochs):                                   # EPOCH = one full pass over the dataset
        order = list(range(len(items))); random.Random(1000 + ep).shuffle(order)
        for si in range(nstep):                                  # STEP = B prompts x K rollouts
            batch = [items[order[si * B + j]] for j in range(B)]
            samples = [sample_of(k, t) for k, t, _ in batch]
            rolls = rollout(samples, a.K)                        # list B, each K (frames, ids)
            # collect all rollouts + GRPO advantages (normalized WITHIN each prompt's K group)
            pairs, advs, step_r = [], [], []
            for bi, (k, t, gt) in enumerate(batch):
                R = np.array([match_reward(fr, gt, a.reward) for fr, _ in rolls[bi]])
                step_r.append(R.mean())
                adv = (R - R.mean()) / (R.std() + 1e-4)
                for i in range(a.K):
                    if abs(adv[i]) < 1e-6 or not rolls[bi][i][1]:
                        continue
                    pairs.append((samples[bi], rolls[bi][i][1])); advs.append(float(adv[i]))
            ntot = max(len(pairs), 1)
            ref_lps = [None] * len(pairs)
            if a.kl > 0 and pairs:                               # SFT-reference log-probs (LoRA+fc swapped)
                cur_lora = [p.detach().clone() for p in lora_params]
                cur_fc = {k: v.detach().clone() for k, v in fc.state_dict().items()}
                for p, r in zip(lora_params, ref_lora):
                    p.data.copy_(r)
                fc.load_state_dict(ref_fc); loz.W.model.eval(); fc.eval()
                with torch.no_grad():
                    ref_lps = []
                    for c0 in range(0, len(pairs), a.lp_chunk):
                        ref_lps.extend([l.detach() for l in logprob_batch(pairs[c0:c0 + a.lp_chunk])])
                for p, c in zip(lora_params, cur_lora):
                    p.data.copy_(c)
                fc.load_state_dict(cur_fc)
            loz.W.model.train(); fc.train(); opt.zero_grad(set_to_none=True)
            for c0 in range(0, len(pairs), a.lp_chunk):           # BATCHED log-prob, chunk fits memory
                cp = pairs[c0:c0 + a.lp_chunk]; ca = advs[c0:c0 + a.lp_chunk]
                cr = ref_lps[c0:c0 + a.lp_chunk]
                lps = logprob_batch(cp)
                loss = 0.0
                for av, lp, rl in zip(ca, lps, cr):
                    loss = loss - (av / ntot) * lp               # policy gradient (GRPO advantage)
                    if rl is not None:                           # k3 KL(policy||SFT), anchors greedy @2
                        loss = loss + a.kl * (torch.exp(rl - lp) - (rl - lp) - 1) / ntot
                loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
            rmean = 0.95 * rmean + 0.05 * float(np.mean(step_r)); gstep += 1
            if gstep % 10 == 0:                                   # heartbeat for live monitoring
                print(f"  [hb ep{ep} step{si+1}/{nstep}] rmean={rmean:.3f} "
                      f"{gstep/(time.time()-t0):.2f} step/s", flush=True)
            if si in eval_at:
                do_eval(f"ep{ep} step{si+1}/{nstep}", gstep, rmean)
        do_eval(f"ep{ep} END", gstep, rmean)
    print(f"[done] {a.run_name} best mAP@2 {best:.2f}", flush=True)


if __name__ == "__main__":
    main()
