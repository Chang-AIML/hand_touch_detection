"""LLM stage-1 (idx-generation) result table: touch/untouch x {no-NMS, NMS, soft-NMS}
x mAP@{0,1,2} on a split. The LLM generates a sparse frame list per (video,type);
NMS barely changes a sparse list -> the 3 rows are ~equal (unlike the dense MS-TCN)."""
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
_COMMON = "/home/chang_noroot/data2/huyanh/Workspace/hand_touch_detection"
sys.path.insert(0, _COMMON)
sys.path.insert(0, os.path.join(_COMMON, "methods/spot_head"))
FEAT = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/feat_interleave"
LAB = os.path.join(_COMMON, "data", "HOI4D-v3")
from common.eval import non_maximum_supression                    # noqa: E402
import eval_nms as en                                             # noqa: E402
from data.questions import GENERIC_Q                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "outputs/idx/idx_multi/best.pt"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()
    en.TOLS = [0, 1, 2]

    from models.wrapper import QwenWrapper
    from models.vjepa_adaptor import VJEPAAdaptor
    from models.idx_localizer import IdxLocalizer
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
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

    vids = json.load(open(os.path.join(LAB, f"{args.split}.json")))
    vids = [v for v in vids if os.path.exists(os.path.join(FEAT, v["video"].replace("/", "__") + ".npy"))]
    truth = [{"video": v["video"], "num_frames": v["num_frames"],
              "events": [{"label": e["label"], "frame": e["frame"]} for e in v["events"]]} for v in vids]
    types = ("touch", "untouch")
    items = [(v, t) for v in vids for t in types if any(e["label"] == t for e in v["events"])]
    pr = {v["video"]: [] for v in vids}
    fc = {}

    def feats(k):
        if k not in fc:
            fc[k] = np.load(os.path.join(FEAT, k + ".npy")).astype(np.float32)
        return fc[k]

    for b0 in range(0, len(items), args.batch_size):
        chunk = items[b0:b0 + args.batch_size]
        samples = []
        for v, t in chunk:
            k = v["video"].replace("/", "__"); f = feats(k); N = f.shape[0]
            samples.append({"feats": torch.from_numpy(f), "question": GENERIC_Q[t], "event_frames": None,
                            "gt": -1, "fps": 15, "type": t, "video_id": k, "num_frames": N,
                            "full_num_frames": N, "anchor_start_sec": 0, "anchor_num_secs": N // 15})
        for (v, t), (fr, sc) in zip(chunk, loz.predict_multievent_batch(samples)):
            N = v["num_frames"]
            for a, b in zip(fr, sc):
                if 0 <= a < N:
                    pr[v["video"]].append({"label": t, "frame": int(a), "score": float(b)})
        if (b0 // args.batch_size) % 10 == 0:
            print(f"  {b0}/{len(items)}", flush=True)

    pred_hr = [{"video": v["video"], "events": list(pr[v["video"]])} for v in vids]
    modes = [("without NMS", pred_hr),
             ("NMS (w=1)", non_maximum_supression(pred_hr, 1)),
             ("Soft-NMS (w=4,s=0.5)", en.soft_nms(pred_hr, 4, 0.5))]
    hdr = f'{"method":<22} @0     @1     @2    Avg'
    print("\n" + "=" * 46 + f"\nLLM idx-gen (idx_multi) -- {args.split}  mAP (%) [touch+untouch avg]\n" + "=" * 46)
    print(hdr); print("-" * 46)
    for name, pv in modes:
        m = en.maps_quiet(truth, pv)
        print(f"{name:<22} " + " ".join(f"{x:<6.2f}" for x in m) + f"{sum(m)/len(m):.2f}")
    for lab in types:
        print(f"\n--- class: {lab} (AP %) ---"); print(hdr); print("-" * 46)
        for name, pv in modes:
            aps = en.per_class_aps(truth, pv).get(lab, [0, 0, 0])
            print(f"{name:<22} " + " ".join(f"{x:<6.2f}" for x in aps) + f"{sum(aps)/len(aps):.2f}")


if __name__ == "__main__":
    main()
