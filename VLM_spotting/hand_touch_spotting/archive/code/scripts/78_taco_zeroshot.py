"""ZERO-SHOT end-to-end transfer test: run the HOI4D-trained idx-localizer on UNSEEN TACO
(touchmoment C* videos) and compute the real E2E mAP@0/1/2 -- the actual "does my model
transfer" question (the probe only tested representation). For each connector (M1 frozen,
SFT-LoRA) eval on TACO-test AND a size-matched HOI4D-test reference, same code/annotation."""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/home/chang_noroot/data2/hf_cache")
_COMMON = "/home/chang_noroot/data2/huyanh/Workspace/hand_touch_detection"
sys.path.insert(0, _COMMON); sys.path.insert(0, os.path.join(_COMMON, "methods/spot_head"))
TM = "/home/chang_noroot/data2/huyanh/Workspace/dataset/hoi4d/touchmoment"
GRID = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/grid_even"
import eval_nms as en                                             # noqa: E402
from data.questions import GENERIC_Q                              # noqa: E402


def pick(dataset, n=None):
    """dataset 'taco'|'hoi': touchmoment test videos with features."""
    out = []
    for v in json.load(open(os.path.join(TM, "test.json"))):
        vid = v["video"]; isH = vid.startswith("H")
        if (dataset == "hoi") != isH:
            continue
        if not os.path.exists(os.path.join(GRID, vid.replace("/", "__") + ".npy")) or not v["events"]:
            continue
        out.append(v)
    random.Random(0).shuffle(out)
    return out[:n] if n else out


def eval_on(loz, vids):
    truth = [{"video": v["video"], "num_frames": v["num_frames"],
              "events": [{"label": e["label"], "frame": e["frame"]} for e in v["events"]]} for v in vids]
    items = [(v, t) for v in vids for t in ("touch", "untouch") if any(e["label"] == t for e in v["events"])]
    fcache = {}

    def sample_of(v, t):
        k = v["video"].replace("/", "__")
        if k not in fcache:
            fcache[k] = np.load(os.path.join(GRID, k + ".npy"))
        g = fcache[k]; N = min(v["num_frames"], 2 * g.shape[0])
        return {"grid": torch.from_numpy(g), "question": GENERIC_Q[t], "type": t, "video_id": k,
                "fps": 15, "num_frames": N, "full_num_frames": N, "anchor_start_sec": 0,
                "anchor_num_secs": N // 15, "event_frames": None, "gt": -1}

    pr = {v["video"]: [] for v in vids}; ndet = 0
    with torch.no_grad():
        for b0 in range(0, len(items), 8):
            ch = items[b0:b0 + 8]
            for (v, t), (fr, sc) in zip(ch, loz.predict_multievent_batch([sample_of(v, t) for v, t in ch])):
                for f2, s2 in zip(fr, sc):
                    if 0 <= f2 < v["num_frames"]:
                        pr[v["video"]].append({"label": t, "frame": int(f2), "score": float(s2)}); ndet += 1
    m = en.maps_quiet(truth, [{"video": v["video"], "events": pr[v["video"]]} for v in vids])
    ngt = sum(len(v["events"]) for v in vids)
    return m, ndet, ngt


def main():
    dev = "cuda"; en.TOLS = [0, 1, 2]; torch.set_num_threads(8)
    from models.wrapper import QwenWrapper
    from models.vjepa_adaptor import VJEPAAdaptor
    from models.idx_localizer import IdxLocalizer
    from models.frame_compress import FrameCompress

    taco = pick("taco")
    hoi = pick("hoi", n=len(taco))                               # size-matched HOI4D reference
    print(f"[zeroshot] TACO test={len(taco)} vids | HOI4D ref={len(hoi)} vids", flush=True)

    for cshort, cname, ckpt in [("SFT", "SFT-LoRA", "outputs/idx_compress/sft_lora_v2/best.pt"),
                                ("M1", "M1 frozen-compress", "outputs/idx_compress/comp_notext/best.pt")]:
        ck = torch.load(os.path.join(ROOT, ckpt), weights_only=False)
        W = QwenWrapper(device=dev, dtype=torch.bfloat16)
        ad = VJEPAAdaptor(768, W.d_llm, hidden=2048, n_layers=2).to(dev, torch.bfloat16)
        fc = FrameCompress(768, 8, W.d_llm, n_heads=4, stain=False, use_text=False).to(dev, torch.float32)
        fc.set_target_rms_from(W.embed_tokens.weight); fc.load_state_dict(ck["fc"]); fc.eval()
        if "lora" in ck:
            lp = W.add_lora(rank=16, alpha=32, n_layers=W.n_layers, target="all")
            for p, s in zip(lp, ck["lora"]):
                p.data.copy_(s.to(p.device, p.dtype))
        loz = IdxLocalizer(W, ad, use_idx=True, use_anchor=True, anchor_stride=5, anchor_max_side=252,
                           use_mrope=False, fps=15, max_frames=320, grad_checkpoint=False, compress=fc)
        mt, ndt, ngt = eval_on(loz, taco)
        mh, ndh, ngh = eval_on(loz, hoi)
        print(f"\n=== {cname} ===", flush=True)
        print(f"  HOI4D  (in-domain ref) mAP@0/1/2 = {[round(x,2) for x in mh]}  ndet={ndh} ngt={ngh}", flush=True)
        print(f"  TACO   (ZERO-SHOT)     mAP@0/1/2 = {[round(x,2) for x in mt]}  ndet={ndt} ngt={ngt}", flush=True)
        print(f"  -> TACO/HOI4D retention @2 = {mt[2]/max(mh[2],1e-9)*100:.0f}%", flush=True)
        del W, loz, fc, ad; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
