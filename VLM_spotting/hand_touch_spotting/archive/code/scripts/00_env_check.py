"""§1.3 env check -> configs/model_dims.yaml.

Confirms: Qwen3-VL loads and exposes output_hidden_states (tuple len = n_layers+1);
V-JEPA 2.1 ViT-B loads with TUBELET=2; records LLM hidden dim, VJEPA dim, layers, fps.
"""
from __future__ import annotations

import os
import sys

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("HF_HOME", "/home/chang_noroot/data2/hf_cache")


def main():
    from models.wrapper import QwenWrapper
    from data import vjepa_interleave as V

    dev = "cuda"
    W = QwenWrapper(device=dev, dtype=torch.bfloat16)

    # dummy forward -> hidden_states tuple length check
    x = torch.randn(8, W.d_llm, device=dev, dtype=torch.bfloat16)
    hs = W.run(x)
    assert len(hs) == W.n_layers + 1, (len(hs), W.n_layers)

    enc = V.load_encoder(device=dev, dtype=torch.float32)
    n_vjepa_params = sum(p.numel() for p in enc.parameters())

    dims = {
        "llm": {
            "model_id": "Qwen/Qwen3-VL-8B-Instruct",
            "hidden_dim": int(W.d_llm),
            "num_layers": int(W.n_layers),
            "num_hidden_states": int(len(hs)),
            "image_token_id": int(W.model.config.image_token_id),
        },
        "vjepa": {
            "variant": "vjepa2_1_vit_base_384 (native, ema_encoder)",
            "feature_dim": int(V.DIM),
            "img_size": int(V.IMG_SIZE),
            "patch": int(V.PATCH),
            "tubelet": int(V.TUBELET),
            "window": int(V.WIN),
            "grid": int(V.GRID),
            "num_params": int(n_vjepa_params),
        },
        "data": {"dataset": "hoi4d_v3", "fps": 15, "num_frames": 300},
        "sim_layer_candidates": [6, 12, 18, 24, 30, 36],
    }
    out = os.path.join(ROOT, "configs", "model_dims.yaml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    yaml.safe_dump(dims, open(out, "w"), sort_keys=False)
    print("wrote", out)
    print(yaml.safe_dump(dims, sort_keys=False))


if __name__ == "__main__":
    main()
