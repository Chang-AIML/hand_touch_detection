"""Decisive debug (per user's guide): single-sample overfit, full model vs bypass-Qwen.

FULL: adaptor -> inject -> frozen Qwen(L12) -> SimHead cos -> CE. Can it drive ONE
sample's argmax to its GT frame? If not, the Qwen-injection readout is the problem.

BYPASS: adaptor -> normalize -> LOC dot -> CE, NO Qwen. Isolates adaptor/loss/gt.
If BYPASS overfits but FULL doesn't, the frozen LLM isn't routing V-JEPA content
into h_loc (query is content-blind) -> architectural fix needed.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/home/chang_noroot/data2/hf_cache")

from train.train_loop import build, trainable_params      # noqa: E402
from data.dataset import Phase1Dataset                    # noqa: E402


def stats(s, gt):
    s = s.detach().float()
    pred = int(s.argmax())
    return (f"loss_argmax_mae {abs(pred-gt):3d} | pred {pred:3d} gt {gt:3d} | "
            f"s.std {s.std():.3f} s.max {s.max():.3f} s.min {s.min():.3f} "
            f"s[gt] {s[gt]:.3f} rank_gt {int((s > s[gt]).sum())}")


def full_overfit(loz, sample, ps, steps=250, lr=1e-3, temp=None):
    if temp is not None:
        loz.temp = temp
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=0.0)
    gt = sample["gt"]
    for step in range(steps + 1):
        loz.train()
        s_list, _ = loz.forward_batch([sample])
        s = s_list[0]
        loss = F.cross_entropy(s[None], torch.tensor([gt], device=s.device))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ps, 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"  [full] step {step:3d} loss {loss.item():.4f} | {stats(s, gt)}", flush=True)


def bypass_overfit(sample, d_vjepa=768, steps=250, lr=1e-3, temp=0.07):
    dev = "cuda"
    feats = sample["feats"].to(dev).float()
    gt = sample["gt"]
    adaptor = torch.nn.Sequential(torch.nn.LayerNorm(d_vjepa), torch.nn.Linear(d_vjepa, 512),
                                  torch.nn.GELU(), torch.nn.Linear(512, 256)).to(dev)
    loc = torch.nn.Parameter(torch.randn(256, device=dev) * 0.02)
    opt = torch.optim.AdamW(list(adaptor.parameters()) + [loc], lr=lr)
    for step in range(steps + 1):
        h = F.normalize(adaptor(feats), dim=-1)               # (N,256)
        q = F.normalize(loc, dim=-1)
        s = (h @ q) / temp
        loss = F.cross_entropy(s[None], torch.tensor([gt], device=dev))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            print(f"  [bypass] step {step:3d} loss {loss.item():.4f} | {stats(s, gt)}", flush=True)


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs", "train.yaml")))
    ds = Phase1Dataset("train", cfg["feat_dir"])
    W, ad, loc, loz = build(cfg, "cuda")
    ps = trainable_params(ad, loc, loz)

    for si in (0, 5):
        sample = ds[si]
        print(f"\n==== sample {si}: video {sample['video_id']} gt {sample['gt']} "
              f"type {sample['type']} q='{sample['question'][:40]}' ====")
        print("--- BYPASS-Qwen (adaptor->loc, no LLM) ---")
        bypass_overfit(sample, steps=200, lr=1e-3, temp=0.07)
        print("--- FULL model (Qwen L12 + SimHead), temp 0.07 ---")
        # re-build trainables fresh per sample to avoid cross-contamination
        from models.vjepa_adaptor import VJEPAAdaptor
        from models.loc_tokens import LocTokens
        from models.sim_head import SimHead
        ad2 = VJEPAAdaptor(768, W.d_llm, hidden=cfg["adaptor_hidden"]).to("cuda", torch.bfloat16)
        ad2.set_target_rms_from(W.embed_tokens.weight)
        loc2 = LocTokens(W.d_llm, 1).to("cuda", torch.bfloat16); loc2.init_from_embeddings(W.embed_tokens.weight)
        head2 = SimHead(W.d_llm, cfg["sim_proj"]).to("cuda")
        loz.adaptor, loz.loc, loz.sim_head = ad2, loc2, head2
        ps2 = list(ad2.parameters()) + [loc2.loc] + list(head2.parameters())
        full_overfit(loz, sample, ps2, steps=200, lr=1e-3, temp=0.07)


if __name__ == "__main__":
    main()
