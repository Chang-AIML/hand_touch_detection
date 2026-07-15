"""Interpretability: project the trained query-compression per-frame tokens (M1, LLM-
embedding space, RMS-matched) onto the LLM vocabulary — nearest words by cosine. Compare
CONTACT frames (GT touch/untouch) vs NON-CONTACT frames. If contact tokens land on
touch/hand/grasp-like words and non-contact don't, the compressed tokens carry real
(interpretable) semantics. Caveat: tokens were trained for idx-copy, not word-alignment."""
from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/home/chang_noroot/data2/hf_cache")
_COMMON = "/home/chang_noroot/data2/huyanh/Workspace/hand_touch_detection"
LAB = os.path.join(_COMMON, "data", "HOI4D-v3")
GRID = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/grid_even"


def main():
    dev = "cuda"; torch.set_num_threads(8)
    from models.wrapper import QwenWrapper
    from models.frame_compress import FrameCompress
    W = QwenWrapper(device=dev, dtype=torch.bfloat16)
    fc = FrameCompress(768, 8, W.d_llm, n_heads=4, stain=False, use_text=False).to(dev, torch.float32)
    fc.set_target_rms_from(W.embed_tokens.weight)
    fc.load_state_dict(torch.load(os.path.join(ROOT, "outputs/idx_compress/comp_notext/best.pt"),
                                  weights_only=False)["fc"]); fc.eval()
    embW = W.embed_tokens.weight.detach().float()                # (V, d)
    embN = torch.nn.functional.normalize(embW, dim=-1)
    tok = W.tokenizer

    def nearest(vecs, topk=5):                                   # vecs (n,d) -> list of [words]
        v = torch.nn.functional.normalize(vecs.float(), dim=-1)
        sims = v @ embN.t()                                     # (n, V)
        idx = sims.topk(topk, dim=-1).indices.cpu().numpy()
        return [[tok.decode([int(i)]).strip() for i in row] for row in idx]

    vids = [v for v in json.load(open(os.path.join(LAB, "train.json")))
            if os.path.exists(os.path.join(GRID, v["video"].replace("/", "__") + ".npy")) and len(v["events"]) >= 2]
    random.Random(0).shuffle(vids); vids = vids[:60]
    contact_words, noncontact_words = Counter(), Counter()
    c_top1, nc_top1 = Counter(), Counter()
    with torch.no_grad():
        for v in vids:
            k = v["video"].replace("/", "__"); g = np.load(os.path.join(GRID, k + ".npy"))
            N = min(v["num_frames"], 2 * g.shape[0])
            motion = fc(torch.from_numpy(g), torch.zeros(1, W.d_llm), N).float()   # (N, d) LLM-space
            gts = sorted(set(e["frame"] for e in v["events"] if e["frame"] < N))
            noncon = [f for f in range(N) if all(abs(f - g2) > 15 for g2 in gts)]
            random.Random(1).shuffle(noncon); noncon = noncon[:len(gts)]           # balance counts
            for group, frames in (("c", gts), ("nc", noncon)):
                if not frames:
                    continue
                words = nearest(motion[frames])
                for wl in words:
                    (contact_words if group == "c" else noncontact_words).update(wl)
                    (c_top1 if group == "c" else nc_top1).update([wl[0]])
    print("\n=== NEAREST WORDS to compressed V-JEPA tokens (M1) ===")
    print(f"\n[CONTACT frames] top-15 nearest words (top-5/frame pooled):")
    print("  " + ", ".join(f"{w}({n})" for w, n in contact_words.most_common(15)))
    print(f"\n[NON-CONTACT frames] top-15 nearest words:")
    print("  " + ", ".join(f"{w}({n})" for w, n in noncontact_words.most_common(15)))
    print(f"\n[CONTACT top-1 only] {', '.join(f'{w}({n})' for w,n in c_top1.most_common(10))}")
    print(f"[NON-CONTACT top-1 only] {', '.join(f'{w}({n})' for w,n in nc_top1.most_common(10))}")
    # distinctness: words that are much more common for contact than non-contact
    diff = {w: contact_words[w] - noncontact_words[w] for w in set(contact_words) | set(noncontact_words)}
    print(f"\n[MORE contact-specific] {', '.join(f'{w}(+{d})' for w,d in sorted(diff.items(),key=lambda x:-x[1])[:12])}")
    print(f"[MORE noncontact-specific] {', '.join(f'{w}({d})' for w,d in sorted(diff.items(),key=lambda x:x[1])[:12])}")


if __name__ == "__main__":
    main()
