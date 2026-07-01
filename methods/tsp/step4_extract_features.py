#!/usr/bin/env python3
"""Extract per-frame TSP features the E2E-Spot way: dense stride-1 overlapping
sliding window (window = training clip-len = 12), one 512-d feature per frame.

Output: one <video>.npy of shape [num_frames, 512] per video, frame-aligned
(feature[t] = backbone(frames centered on t)), no padding baked in -- exactly
what spot/dataset/feature.py expects.

Only the R(2+1)D-34 backbone is used (GVF/head are pretraining-only). Each frame
is decoded + spatially transformed ONCE; only the backbone forward is dense (12x).
"""
import os, sys, glob, json, argparse
import numpy as np
import torch
import torchvision
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config                              # noqa: E402
config.add_proj_to_path()
from common import transforms as T          # noqa: E402
from models.model import Model              # noqa: E402

FRAMES_DIR = config.FRAMES_DIR
LABEL_DIR  = config.LABEL_DIR                            # train/val/test.json
CLIP_LEN   = config.CLIP_LEN                             # window = training clip-len
HALF       = CLIP_LEN // 2

_normalize = T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                         std=[0.22803, 0.22145, 0.216989])
# valid-style spatial transform (deterministic CenterCrop), identical to training val
_transform = torchvision.transforms.Compose([
    T.ToFloatTensorInZeroOne(), T.Resize((128, 171)), _normalize, T.CenterCrop((112, 112))])


def list_videos():
    vids, seen = [], set()
    for sp in ['train', 'val', 'test']:
        for x in json.load(open(os.path.join(LABEL_DIR, f'{sp}.json'))):
            v = x['video']
            if v not in seen and os.path.isdir(os.path.join(FRAMES_DIR, v)):
                seen.add(v); vids.append(v)
    return vids


def load_video_tensor(vname):
    """Decode + transform all frames once, replicate-pad +-HALF along time.
    Returns [C, n+CLIP_LEN, 112, 112]; window for frame t = clip[:, t:t+CLIP_LEN]."""
    files = sorted(glob.glob(os.path.join(FRAMES_DIR, vname, '*.jpg')))
    n = len(files)
    arr = np.stack([np.asarray(Image.open(f).convert('RGB')) for f in files])  # [n,H,W,C]
    # replicate-pad ends so the window stays centered at boundaries
    pad = np.concatenate([arr[:1]] * HALF + [arr] + [arr[-1:]] * HALF, axis=0)  # [n+CLIP_LEN,...]
    clip = _transform(torch.from_numpy(pad))    # [C, n+CLIP_LEN, 112, 112]
    return clip, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=os.path.join(config.TRAIN_OUT, 'epoch_4.pth'),
                    help='best-by-F1 checkpoint (.pth)')
    ap.add_argument('--out-dir', default=config.FEATURES_OUT)
    ap.add_argument('--batch-size', type=int, default=128, help='windows per forward')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    videos = list_videos()
    if args.limit:
        videos = videos[:args.limit]
    done = {os.path.splitext(f)[0] for f in os.listdir(args.out_dir) if f.endswith('.npy')}
    todo = [v for v in videos if v not in done]
    print(f'videos: {len(videos)}; already done: {len(done)}; to do: {len(todo)}', flush=True)
    if not todo:
        print('nothing to do'); return

    # backbone only (GVF/heads irrelevant for feature extraction). Build the dual-head
    # arch so the checkpoint's keys line up; strict=False tolerates head differences.
    model = Model(backbone='r2plus1d_34', num_classes=[2, 2], num_heads=2,
                  concat_gvf=True, gvf_size=config.GVF_DIM, progress=False)
    sd = torch.load(args.ckpt, map_location='cpu', weights_only=False); sd = sd.get('model', sd)
    model.load_state_dict(sd, strict=False)
    backbone = model.features.to(args.device).eval()
    print(f'loaded backbone from {args.ckpt}', flush=True)

    with torch.no_grad():
        for vi, vname in enumerate(todo):
            clip, n = load_video_tensor(vname)            # [C, n+CLIP_LEN, ...]
            clip = clip.to(args.device, non_blocking=True)
            feats = np.empty((n, 512), dtype=np.float32)
            for s in range(0, n, args.batch_size):
                e = min(s + args.batch_size, n)
                # window for frame t = clip[:, t:t+CLIP_LEN]
                batch = torch.stack([clip[:, t:t + CLIP_LEN] for t in range(s, e)])  # [b,C,12,112,112]
                with torch.autocast('cuda', dtype=torch.float16):
                    f = backbone(batch)                   # [b, 512]
                feats[s:e] = f.float().cpu().numpy()
            np.save(os.path.join(args.out_dir, f'{vname}.npy'), feats)
            if (vi + 1) % 50 == 0 or vi == 0:
                print(f'  [{vi+1}/{len(todo)}] {vname}: {feats.shape}', flush=True)
            del clip
    print(f'DONE. wrote {len(todo)} feature files -> {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
