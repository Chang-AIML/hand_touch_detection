"""On-the-fly frozen V-JEPA 2.1 ViT-B even-pass grid extractor (no disk).

Produces the even-pass spatial-token grid the connector was trained on, computed inside a
forward pass from RGB frames. BIT-FAITHFUL-to-offline-cache contract (do not change):
  - encoder always fp32 (native RoPE requires it; matches the offline cache)
  - even pass only: token k covers tubelet(2k,2k+1); return the first ceil(T/2) tokens
  - NO spatial pool, NO interpolation here (FrameCompress interpolates ceil(T/2)->T downstream)
"""
from __future__ import annotations
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dpc.paths import VJEPA_REPO, VJEPA_CKPT

# V-JEPA 2.1 ViT-B constants (native repo vjepa2_1_vit_base_384)
IMG_SIZE, PATCH, TUBELET = 384, 16, 2
GRID = IMG_SIZE // PATCH                    # 24 -> 576 spatial tokens per temporal token
DIM = 768                                   # ViT-B embed dim
WIN = 16                                    # frames per encoder window (multiple of TUBELET)
TOK_T = WIN // TUBELET                      # temporal tokens per window
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_encoder(repo, ckpt, device="cuda", dtype=torch.float32):
    """Frozen V-JEPA 2.1 ViT-B encoder from the local ckpt; eval, requires_grad=False."""
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from src.hub.backbones import vjepa2_1_vit_base_384, _clean_backbone_key
    encoder, _ = vjepa2_1_vit_base_384(pretrained=False)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    missing, _ = encoder.load_state_dict(_clean_backbone_key(sd["ema_encoder"]), strict=False)
    hard = [k for k in missing if "pos_embed" not in k]   # RoPE has no pos_embed; nothing else may miss
    assert not hard, f"unexpected missing keys: {hard[:8]}"
    return encoder.to(device, dtype).eval().requires_grad_(False)


def preprocess_frames(frames_rgb):
    """List of HxWx3 uint8 RGB -> (N,3,384,384) ImageNet-normed: shortest-edge resize 384 + center crop."""
    arr = np.stack(frames_rgb).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(0, 3, 1, 2)
    h, w = t.shape[-2:]
    scale = IMG_SIZE / min(h, w)
    t = F.interpolate(t, size=(round(h * scale), round(w * scale)), mode="bilinear", align_corners=False)
    top, left = (t.shape[-2] - IMG_SIZE) // 2, (t.shape[-1] - IMG_SIZE) // 2
    t = t[:, :, top:top + IMG_SIZE, left:left + IMG_SIZE]
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (t - mean) / std


def _pad_to_multiple(x, m):
    """Pad axis-0 up to a multiple of m by repeating the last frame."""
    pad = (-x.shape[0]) % m
    return torch.cat([x, x[-1:].expand(pad, *x.shape[1:])], dim=0) if pad else x


def _even_pass_grid(proc, encoder, device, batch_windows):
    """proc (M,3,384,384) -> (nwin*TOK_T, 576, 768) unpooled per-temporal-token spatial grid (fp32 numpy)."""
    proc = _pad_to_multiple(proc, WIN)
    nwin = proc.shape[0] // WIN
    wins = proc.view(nwin, WIN, 3, IMG_SIZE, IMG_SIZE)
    chunks = []
    for s in range(0, nwin, batch_windows):
        b = wins[s:s + batch_windows].to(device, torch.float32, non_blocking=True).permute(0, 2, 1, 3, 4).contiguous()
        with torch.no_grad():
            h = encoder(b)                                       # (bw, TOK_T*576, 768)
        h = h.view(h.shape[0], TOK_T, GRID * GRID, DIM)          # NO spatial pool
        chunks.append(h.float().cpu())
    return torch.cat(chunks, 0).reshape(nwin * TOK_T, GRID * GRID, DIM).numpy()


class OnlineVJEPA(nn.Module):
    """Frozen V-JEPA 2.1 even-pass grid extractor, on the fly.
    `dtype` = RETURN dtype (default bf16); the encoder always runs fp32."""
    def __init__(self, device, dtype=torch.bfloat16, repo=None, ckpt=None, compute_dtype=torch.float32):
        super().__init__()
        self.device = torch.device(device)
        self.out_dtype = dtype
        self.compute_dtype = compute_dtype     # fp32 = bit-match offline cache; bf16 = ~2x faster (autocast)
        self.encoder = load_encoder(repo or VJEPA_REPO, ckpt or VJEPA_CKPT, self.device, torch.float32)

    def train(self, mode=True):                # keep the frozen encoder in eval() even under train()
        super().train(mode); self.encoder.eval(); return self

    @staticmethod
    def _to_frame_list(frames):
        if isinstance(frames, torch.Tensor):
            frames = frames.detach().cpu().numpy()
        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"expected frames (T,H,W,3), got {frames.shape}")
        if frames.dtype != np.uint8:
            frames = frames.astype(np.uint8)
        return [frames[i] for i in range(frames.shape[0])]

    @torch.no_grad()
    def extract_grid(self, frames_uint8, batch_windows=32):
        """RGB window (T,H,W,3) -> even-pass grid (ceil(T/2),576,768) on self.device.
        Encoder fp32 (autocast only when compute_dtype != fp32); no interp (FrameCompress does it)."""
        self.encoder.eval()
        frames = self._to_frame_list(frames_uint8)
        T = len(frames)
        amp = self.compute_dtype != torch.float32
        with torch.autocast(device_type=self.device.type, enabled=amp, dtype=self.compute_dtype):
            proc = preprocess_frames(frames)
            even = _even_pass_grid(proc, self.encoder, self.device, batch_windows)
        even = even[:(T + 1) // 2]              # keep #even-indexed frames, drop pad-tail
        return torch.from_numpy(np.ascontiguousarray(even)).to(self.device, self.out_dtype)
