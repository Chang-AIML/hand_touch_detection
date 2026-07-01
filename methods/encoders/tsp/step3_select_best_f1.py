#!/usr/bin/env python3
"""Evaluate every saved TSP epoch checkpoint on the VALIDATION set and report
foreground (contact) Precision / Recall / F1 -- NOT accuracy, because the
Foreground/Background classes are imbalanced (~1:1.5).

Picks the epoch with the highest Foreground F1 and writes best_by_f1.json.
Mirrors train.py's validation dataset/transforms exactly (frames, clip-len 12,
MViT-B GVF, clips_per_segment=5, temporal_jittering=False -> deterministic).
"""
import os, sys, json, glob, argparse
import numpy as np
import torch
import torchvision

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import config                                            # noqa: E402
config.add_proj_to_path()
from common import transforms as T                       # noqa: E402
from models.model import Model                           # noqa: E402
from frame_untrimmed_video_dataset import FrameUntrimmedVideoDataset  # noqa: E402

FRAMES_DIR = config.FRAMES_DIR
OUTPUT_DIR = config.TRAIN_OUT
GVF        = config.GVF_PATH
VAL_CSV    = config.VAL_CSV
LABEL_COL  = 'temporal-region-label'
LABEL_MAP  = json.load(open(config.TR_LABEL_MAP))        # {'Background':0,'Foreground':1}
FG = LABEL_MAP['Foreground']


def build_val_loader(batch_size, workers):
    normalize = T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                            std=[0.22803, 0.22145, 0.216989])
    tf = torchvision.transforms.Compose([
        T.ToFloatTensorInZeroOne(), T.Resize((128, 171)), normalize, T.CenterCrop((112, 112))])
    ds = FrameUntrimmedVideoDataset(
        csv_filename=VAL_CSV,
        root_dir=FRAMES_DIR, clip_length=config.CLIP_LEN, frame_rate=config.FRAME_RATE, clips_per_segment=5,
        temporal_jittering=False, transforms=tf,
        label_columns=[LABEL_COL], label_mappings=[LABEL_MAP], global_video_features=GVF)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                       num_workers=workers, pin_memory=True)


def load_weights(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = sd.get('model', sd)
    model.load_state_dict(sd)
    return model


@torch.no_grad()
def run(model, loader, device):
    model.eval()
    preds, tgts = [], []
    for sample in loader:
        clip = sample['clip'].to(device, non_blocking=True)
        gvf = sample['gvf'].to(device, non_blocking=True) if 'gvf' in sample else None
        t = sample[LABEL_COL]
        with torch.autocast('cuda', dtype=torch.float16):
            out = model(clip, gvf=gvf)[1]            # region head (fc2, GVF-fed); [B,2]
        preds.append(out.float().argmax(1).cpu().numpy())
        tgts.append(t.numpy())
    return np.concatenate(preds), np.concatenate(tgts)


def prf(preds, tgts):
    tp = int(((preds == FG) & (tgts == FG)).sum())
    fp = int(((preds == FG) & (tgts != FG)).sum())
    fn = int(((preds != FG) & (tgts == FG)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    acc = float((preds == tgts).mean())
    return p, r, f1, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    ckpts = sorted(glob.glob(os.path.join(OUTPUT_DIR, 'epoch_*.pth')),
                   key=lambda p: int(p.split('epoch_')[1].split('.')[0]))
    assert ckpts, f'no epoch_*.pth in {OUTPUT_DIR}'
    print(f'found {len(ckpts)} checkpoints')

    loader = build_val_loader(args.batch_size, args.workers)
    # dual-head checkpoint: head0=action (touch/untouch), head1=region (FG/BG)+GVF.
    # We evaluate the region head for Foreground/Background F1 (see infer(): out[1]).
    model = Model(backbone='r2plus1d_34', num_classes=[2, 2], num_heads=2,
                  concat_gvf=True, gvf_size=config.GVF_DIM, progress=False).to(args.device)

    rows, best = [], None
    print(f'\n{"epoch":>6} {"FG-P":>7} {"FG-R":>7} {"FG-F1":>7} {"acc":>7}')
    for c in ckpts:
        ep = int(c.split('epoch_')[1].split('.')[0])
        load_weights(model, c)
        preds, tgts = run(model, loader, args.device)
        p, r, f1, acc = prf(preds, tgts)
        rows.append({'epoch': ep, 'ckpt': c, 'fg_precision': p, 'fg_recall': r,
                     'fg_f1': f1, 'accuracy': acc})
        print(f'{ep:>6} {p:>7.4f} {r:>7.4f} {f1:>7.4f} {acc:>7.4f}')
        if best is None or f1 > best['fg_f1']:
            best = rows[-1]

    out = {'best_by_f1': best, 'all': rows}
    with open(os.path.join(OUTPUT_DIR, 'best_by_f1.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nBEST by Foreground-F1: epoch {best["epoch"]} '
          f'(F1={best["fg_f1"]:.4f}, P={best["fg_precision"]:.4f}, R={best["fg_recall"]:.4f})')
    print(f'-> {best["ckpt"]}')
    print(f'written {os.path.join(OUTPUT_DIR, "best_by_f1.json")}')


if __name__ == '__main__':
    main()
