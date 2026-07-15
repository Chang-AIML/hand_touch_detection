"""Tier-2: the FrameCompress checkpoint contract. Every existing connector ckpt must load
strict=True into the refactored FrameCompress, and the key set must be exactly the frozen 13."""
import sys, glob, torch
sys.path.insert(0, ".")
from models.frame_compress import FrameCompress

FROZEN_KEYS = {
    "q", "in_ln.weight", "in_ln.bias",
    "a3.n.weight", "a3.n.bias",
    "a3.a.in_proj_weight", "a3.a.in_proj_bias", "a3.a.out_proj.weight", "a3.a.out_proj.bias",
    "out.0.weight", "out.0.bias", "out.1.weight", "out.1.bias",
}
CKPTS = sorted(glob.glob("outputs/local_eval/conn_s750.pt") + glob.glob("outputs/local_eval/conn_MIX*.pt"))
assert CKPTS, "no connector ckpts found"

fc = FrameCompress(768, n_q=8, d_llm=4096, n_heads=4)
model_keys = set(fc.state_dict().keys())
assert model_keys == FROZEN_KEYS, f"module key set drift:\n  extra={model_keys-FROZEN_KEYS}\n  missing={FROZEN_KEYS-model_keys}"

n = sum(p.numel() for p in fc.parameters())
assert 27_500_000 < n < 27_600_000, f"param count {n:,} off (~27.55M expected)"

for c in CKPTS:
    sd = torch.load(c, map_location="cpu", weights_only=False)["fc"]
    assert set(sd.keys()) == FROZEN_KEYS, f"{c}: ckpt keys != frozen set: {set(sd)^FROZEN_KEYS}"
    miss, unexp = fc.load_state_dict(sd, strict=True)  # raises if mismatch
    print(f"  OK strict-load {c.split('/')[-1]}")
print(f"STATE_DICT PASS: {len(CKPTS)} ckpts load strict; {len(FROZEN_KEYS)} keys; {n:,} params")
