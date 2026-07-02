"""Per-frame feature extraction with V-JEPA 2.1 (ViT-L/16, 384, dist from ViT-G).

V-JEPA 2.1 is the 2026-03 release tuned for high-quality, temporally-consistent
DENSE features -- a good fit for per-frame extraction. It still uses tubelet_size=2
(a 16-frame clip -> 8 temporal tokens), so to get a feature per frame we use the
even/odd offset trick:

    even pass:  frames [0,1,2,...]   -> tubelet tokens -> frames 0,2,4,...
    odd  pass:  frames [1,2,3,...]   -> tubelet tokens -> frames 1,3,5,...

Per the goal we do NOT interleave. We save TWO temporal-only files per clip:

    <clip>_even.npy  (Ne, 1024)   Ne = ceil(N/2),  rows align to frames 0,2,4,...
    <clip>_odd.npy   (No, 1024)   No = floor(N/2),  rows align to frames 1,3,5,...

To reconstruct a per-frame (N,1024) sequence later:
    out = np.empty((N,1024)); out[0::2]=even; out[1::2]=odd

Model: built from the local repo (app.vjepa_2_1), encoder weights = checkpoint's
"ema_encoder". Spatial tokens (24x24) are mean-pooled -> temporal-only per frame.

Env: conda `vjepa21`. Run one process per GPU (see run_dual_gpu.sh).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TT

import frame_io

VJEPA_REPO = os.environ.get("TOUCH_VJEPA_REPO", "/data/dong/project/Workspace/repos/vjepa2")
CKPT_DIR = os.environ.get("TOUCH_VJEPA_CKPT_DIR", "/data/dong/project/Workspace/repos/feature_extraction/ckpts")

# model_name -> (hub builder, checkpoint filename). "base" is the lightest V-JEPA 2.1.
MODELS = {
    "base": ("vjepa2_1_vit_base_384", "vjepa2_1_vitb_dist_vitG_384.pt"),
    "large": ("vjepa2_1_vit_large_384", "vjepa2_1_vitl_dist_vitG_384.pt"),
}

IMG_SIZE = 384
PATCH = 16
TUBELET = 2
WIN = 16                       # frames per window (must be even)
TOK_T = WIN // TUBELET         # 8 temporal tokens / window
GRID = IMG_SIZE // PATCH       # 24x24 spatial tokens
IN1K_MEAN = (0.485, 0.456, 0.406)
IN1K_STD = (0.229, 0.224, 0.225)


def build_encoder(model: str, ckpt_path: str, device):
    sys.path.insert(0, VJEPA_REPO)
    import src.hub.backbones as bb

    builder = getattr(bb, MODELS[model][0])
    encoder, _predictor = builder(pretrained=False)                  # no hub download
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    enc_sd = bb._clean_backbone_key(state["ema_encoder"])            # checkpoint_key for 2.1 dist
    encoder.load_state_dict(enc_sd, strict=True)                     # RoPE -> pos_embed unused
    # fp32: RoPE upcasts q/k to float internally, so half() would clash with half v in SDPA.
    return encoder.to(device).eval()


def build_transform():
    return TT.Compose([
        TT.Resize(IMG_SIZE),            # shortest side -> 384
        TT.CenterCrop(IMG_SIZE),        # 384 x 384
        TT.ToTensor(),                  # CHW in [0,1]
        TT.Normalize(IN1K_MEAN, IN1K_STD),
    ])


def _pad_to_multiple(x: torch.Tensor, m: int) -> torch.Tensor:
    pad = (-x.shape[0]) % m
    if pad:
        x = torch.cat([x, x[-1:].expand(pad, *x.shape[1:])], dim=0)
    return x


class ClipWindowDataset(torch.utils.data.Dataset):
    """One item = all 16-frame windows of one clip's offset stream, preprocessed on a
    worker process. The consumer batches windows ACROSS clips to fill the GPU.

        even stream: frames [0,1,2,...]  -> windows -> tubelet tokens -> frames 0,2,4,...
        odd  stream: frames [1,2,3,...]  -> windows -> tubelet tokens -> frames 1,3,5,...
    """

    def __init__(self, clips, which, transform, out_dir, overwrite):
        self.clips, self.which, self.transform = clips, which, transform
        self.out_dir, self.overwrite = out_dir, overwrite

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        clip_id, clip_dir = self.clips[i]
        out_path = os.path.join(self.out_dir, f"{clip_id}_{self.which}.npy")
        if not self.overwrite and os.path.exists(out_path):
            return None                                   # resume: already done
        paths = frame_io.list_frame_paths(clip_dir)
        n = len(paths)
        proc = torch.stack([self.transform(Image.open(p).convert("RGB")) for p in paths])  # (n,3,384,384)
        if self.which == "even":
            seq, keep = proc, (n + 1) // 2                # rows align to frames 0,2,4,...
        else:
            seq, keep = proc[1:], n // 2                  # rows align to frames 1,3,5,...
        seq = _pad_to_multiple(seq, WIN)
        nwin = seq.shape[0] // WIN
        wins = seq.view(nwin, WIN, 3, IMG_SIZE, IMG_SIZE).permute(0, 2, 1, 3, 4).contiguous()  # (nwin,3,16,384,384)
        return {"clip_id": clip_id, "wins": wins, "keep": keep, "out_path": out_path}


@torch.no_grad()
def run_stream(clips, which, encoder, transform, device, batch_windows, workers, out_dir, overwrite):
    """Cross-clip streaming: accumulate windows from many clips up to `batch_windows`,
    run one encoder forward per full batch, scatter the 8 temporal tokens/window back to
    each clip, and save (keep, D) temporal-only features as each clip completes."""
    ds = ClipWindowDataset(clips, which, transform, out_dir, overwrite)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, num_workers=workers, prefetch_factor=2 if workers else None,
        collate_fn=lambda b: b[0])

    buf_w, buf_map = [], []                               # windows + (clip_id, window_idx)
    pending = {}                                          # clip_id -> {parts, keep, out_path, left}
    t0 = time.time()
    done = [0]

    def flush():
        if not buf_w:
            return
        x = torch.cat(buf_w, 0).to(device, torch.float32, non_blocking=True)   # (B,3,16,384,384)
        h = encoder(x)                                                         # (B, 8*576, D)
        h = h.view(h.shape[0], TOK_T, GRID * GRID, h.shape[-1]).mean(2).cpu()  # (B, 8, D)
        for (cid, widx), feat in zip(buf_map, h):
            st = pending[cid]
            st["parts"][widx] = feat                      # (8, D)
            st["left"] -= 1
            if st["left"] == 0:
                seq = torch.cat(st["parts"], 0)[: st["keep"]].numpy().astype(np.float16)  # (keep, D)
                np.save(st["out_path"], seq)
                del pending[cid]
                done[0] += 1
                if done[0] % 50 == 0 or done[0] == 1:
                    print(f"  [{done[0]}/{len(clips)}] {cid} {seq.shape} "
                          f"| {(time.time()-t0)/done[0]:.2f}s/clip", flush=True)
        buf_w.clear(); buf_map.clear()

    for item in loader:
        if item is None:
            continue
        cid, wins, keep, out_path = item["clip_id"], item["wins"], item["keep"], item["out_path"]
        nwin = wins.shape[0]
        pending[cid] = {"parts": [None] * nwin, "keep": keep, "out_path": out_path, "left": nwin}
        for j in range(nwin):
            buf_w.append(wins[j:j + 1]); buf_map.append((cid, j))
            if len(buf_w) >= batch_windows:
                flush()
    flush()                                               # leftover tail
    print(f"[vjepa2.1] pass={which} done: {done[0]} new clips in {(time.time()-t0)/60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", choices=list(MODELS), default="large", help="V-JEPA 2.1 size")
    ap.add_argument("--ckpt", default=None, help="override checkpoint path (default: from --model)")
    ap.add_argument("--pass", dest="which", choices=["even", "odd"], default="even",
                    help="even/odd offset stream to compute (run one GPU per stream)")
    ap.add_argument("--batch-windows", type=int, default=96,
                    help="cross-clip GPU batch (windows fused across clips to fill the GPU)")
    ap.add_argument("--workers", type=int, default=8, help="parallel frame-loading workers")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--video-list", default=None,
                    help="txt of video ids (one per line); restrict extraction to these clips")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = args.ckpt or os.path.join(CKPT_DIR, MODELS[args.model][1])
    assert os.path.exists(ckpt), f"checkpoint not found: {ckpt}"
    encoder = build_encoder(args.model, ckpt, device)
    transform = build_transform()
    os.makedirs(args.out_dir, exist_ok=True)

    clips = frame_io.list_clips(args.frames_dir)
    if args.video_list:
        keep = {l.strip() for l in open(args.video_list) if l.strip()}
        clips = [c for c in clips if c[0] in keep]
    clips = frame_io.shard(clips, args.shard_id, args.num_shards)
    if args.limit:
        clips = clips[: args.limit]
    print(f"[vjepa2.1-{args.model}] {len(clips)} clips | pass={args.which} | bw={args.batch_windows} | "
          f"workers={args.workers} | shard {args.shard_id+1}/{args.num_shards} | "
          f"device={args.device} | out={args.out_dir}", flush=True)

    run_stream(clips, args.which, encoder, transform, device,
               args.batch_windows, args.workers, args.out_dir, args.overwrite)


if __name__ == "__main__":
    main()
