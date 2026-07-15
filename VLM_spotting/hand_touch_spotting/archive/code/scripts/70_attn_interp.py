"""Interpretability that actually fits idx-trained tokens: WHERE do the compression's
8 queries attend on the 576-patch (24x24) grid? Overlay the query-attention heatmap on
the real frame, for CONTACT vs NON-CONTACT frames. If contact frames focus on the
hand/contact region, the connector learned meaningful spatial semantics."""
from __future__ import annotations

import glob
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
LAB = os.path.join(_COMMON, "data", "HOI4D-v3")
GRID = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/grid_even"
FRAMES = "/home/chang_noroot/data2/huyanh/Workspace/dataset/hoi4d/frames"


def main():
    dev = "cuda"; torch.set_num_threads(8)
    from models.wrapper import QwenWrapper
    from models.frame_compress import FrameCompress
    W = QwenWrapper(device=dev, dtype=torch.bfloat16)
    fc = FrameCompress(768, 8, W.d_llm, n_heads=4, stain=False, use_text=False).to(dev, torch.float32)
    fc.set_target_rms_from(W.embed_tokens.weight)
    fc.load_state_dict(torch.load(os.path.join(ROOT, "outputs/idx_compress/sft_lora_v2/best.pt"),
                                  weights_only=False)["fc"]); fc.eval()

    @torch.no_grad()
    def attn_map(even_grid, N, frame):
        """24x24 attention of the 8 queries over patches at `frame`."""
        grid = fc._interp(even_grid.to(dev).float(), N)          # (N,576,768)
        p = fc.in_ln(grid[frame:frame + 1])                      # (1,576,d)
        q = fc.q.unsqueeze(0)                                    # (1,8,d)
        _, w = fc.a3.a(fc.a3.n(q), p, p, need_weights=True, average_attn_weights=True)  # (1,8,576)
        a = w[0].mean(0).reshape(24, 24).cpu().numpy()          # avg over 8 queries
        return a

    vids = [v for v in json.load(open(os.path.join(LAB, "val.json")))
            if os.path.isdir(os.path.join(FRAMES, v["video"])) and len(v["events"]) >= 2][:8]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    os.makedirs(os.path.join(ROOT, "plot"), exist_ok=True)
    fig, axes = plt.subplots(len(vids), 2, figsize=(6, 3 * len(vids)))
    # numeric summary: attention concentration (max weight) contact vs non-contact
    conc_c, conc_nc = [], []
    for r, v in enumerate(vids):
        k = v["video"].replace("/", "__"); g = np.load(os.path.join(GRID, k + ".npy"))
        N = min(v["num_frames"], 2 * g.shape[0]); eg = torch.from_numpy(g)
        gts = sorted(e["frame"] for e in v["events"] if e["frame"] < N)
        cf = gts[len(gts) // 2]                                  # a contact frame
        ncf = next((f for f in range(N) if all(abs(f - x) > 20 for x in gts)), 0)
        frs = sorted(glob.glob(os.path.join(FRAMES, v["video"], "*.jpg")))
        for c, (fr, tag) in enumerate([(cf, "CONTACT"), (ncf, "non-contact")]):
            a = attn_map(eg, N, fr)
            (conc_c if c == 0 else conc_nc).append(float(a.max() / (a.mean() + 1e-9)))
            img = np.asarray(Image.open(frs[min(fr, len(frs) - 1)]).convert("RGB").resize((384, 384)))
            ax = axes[r, c] if len(vids) > 1 else axes[c]
            ax.imshow(img); hm = np.kron(a, np.ones((16, 16)))   # 24x24 -> 384x384
            ax.imshow(hm, cmap="jet", alpha=0.45, extent=(0, 384, 384, 0))
            ax.set_title(f"{tag} f{fr}", fontsize=8); ax.axis("off")
    out = os.path.join(ROOT, "plot", "attn_interp_sft.png")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    print(f"[saved] {out}", flush=True)
    print(f"attention concentration (max/mean): CONTACT {np.mean(conc_c):.2f} | "
          f"non-contact {np.mean(conc_nc):.2f}", flush=True)


if __name__ == "__main__":
    main()
