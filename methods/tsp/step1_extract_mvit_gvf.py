#!/usr/bin/env python3
"""Extract Global Video Features (GVF) using MViT-B (16x4, Kinetics-400) per
E2E-Spot paper sec. B.3, reading JPG frames directly (no mp4/ffmpeg).

Paper B.3: "For the pre-trained global video feature (GVF), we use pre-extracted
MViT-B features ... as these serve a similar function to the frozen GVF in the
original implementation."

Pipeline:
  - encoder = torchvision mvit_v1_b(KINETICS400_V1)  (== MViT-B 16x4), head removed -> 768-d
  - clip = 16 frames sampled at temporal stride 4 (spans 64 frames), non-overlapping
  - preprocessing = resize short side 256 -> center crop 224 -> Kinetics norm (0.45/0.225)
  - GVF = max-pool of all clip features  ->  one 768-d vector per video
Writes an h5 keyed by video name (no extension), matching what the TSP
UntrimmedVideoDataset expects: f[basename(filename).split('.')[0]].
"""
import os, sys, glob, json, argparse
import numpy as np
import torch
import torch.nn as nn
import h5py
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import mvit_v1_b, MViT_V1_B_Weights

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config                                   # noqa: E402
FRAMES_DIR = config.FRAMES_DIR
LABEL_DIR  = config.LABEL_DIR                    # train/val/test.json live here
CLIP_T, FRAME_STRIDE = 16, 4          # MViT-B "16x4"
SPAN = CLIP_T * FRAME_STRIDE          # 64 frames covered per clip
STEP = SPAN                           # non-overlapping clips
MEAN = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
STD  = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)


def preprocess(vid_uint8):
    """[T,H,W,C] uint8 -> [C,T,H,W] float, resized/cropped/normalized for MViT-B."""
    x = vid_uint8.permute(3, 0, 1, 2).float().div_(255.0)   # [C,T,H,W]
    x = TF.resize(x, [256], antialias=True)                 # short side -> 256
    x = TF.center_crop(x, [224, 224])
    x = (x - MEAN) / STD
    return x


def list_videos():
    vids = []
    for sp in ['train', 'val', 'test']:
        for x in json.load(open(os.path.join(LABEL_DIR, f'{sp}.json'))):
            vids.append(x['video'])
    seen, out = set(), []
    for v in vids:
        if v not in seen and os.path.isdir(os.path.join(FRAMES_DIR, v)):
            seen.add(v); out.append(v)
    return out


def clip_starts(n):
    if n < SPAN:
        return [0]                       # single clip, padded by repeating last frame
    return list(range(0, n - SPAN + 1, STEP))


class ClipDataset(Dataset):
    """One item = one clip. Returns (video_idx, clip[C,T,H,W])."""
    def __init__(self, videos):
        self.frames = {v: sorted(glob.glob(os.path.join(FRAMES_DIR, v, '*.jpg')))
                       for v in videos}
        self.videos = videos
        self.index = []
        for vi, v in enumerate(videos):
            for s in clip_starts(len(self.frames[v])):
                self.index.append((vi, s))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        vi, s = self.index[i]
        files = self.frames[self.videos[vi]]
        n = len(files)
        idxs = [min(s + k * FRAME_STRIDE, n - 1) for k in range(CLIP_T)]
        frames = [np.asarray(Image.open(files[j]).convert('RGB')) for j in idxs]
        vid = torch.from_numpy(np.stack(frames))             # [T,H,W,C] uint8
        return vi, preprocess(vid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=config.GVF_PATH)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--limit', type=int, default=0, help='only first N videos (debug)')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    videos = list_videos()
    if args.limit:
        videos = videos[:args.limit]
    print(f'episodes found: {len(videos)}', flush=True)

    done = set()
    if os.path.exists(args.out):
        with h5py.File(args.out, 'r') as f:
            done = set(f.keys())
    todo = [v for v in videos if v not in done]
    print(f'already done: {len(done)}; to process: {len(todo)}', flush=True)
    if not todo:
        print('nothing to do'); return

    print('loading MViT-B (16x4, Kinetics-400)...', flush=True)
    model = mvit_v1_b(weights=MViT_V1_B_Weights.KINETICS400_V1)
    model.head = nn.Identity()                # -> 768-d pooled cls feature
    model.to(args.device).eval()

    ds = ClipDataset(todo)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
    print(f'total clips: {len(ds)}', flush=True)

    feats = {vi: [] for vi in range(len(todo))}
    seen = 0
    with torch.no_grad():
        for vis, clips in dl:
            clips = clips.to(args.device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16):
                f = model(clips)                        # [B, 768]
            f = f.float().cpu().numpy()
            for k, vi in enumerate(vis.numpy()):
                feats[int(vi)].append(f[k])
            seen += len(vis)
            if seen % (args.batch_size * 20) < args.batch_size:
                print(f'  {seen}/{len(ds)} clips', flush=True)

    with h5py.File(args.out, 'a') as h:
        for vi, v in enumerate(todo):
            arr = np.stack(feats[vi])                    # [num_clips, 768]
            gvf = arr.max(axis=0).astype(np.float32)     # [768]
            if v in h:
                del h[v]
            h.create_dataset(v, data=gvf)
    print(f'DONE. wrote MViT-B GVF for {len(todo)} videos -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
