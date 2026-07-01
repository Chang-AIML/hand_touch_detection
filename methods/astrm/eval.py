#!/usr/bin/env python3
""" Standalone evaluation for a trained ASTRM checkpoint.

Loads a checkpoint from a training run dir and reports mAP@{0,1,2,4} on the
chosen split for all three post-processings: no-NMS / hard-NMS / soft-NMS.
The model config is read back from <save_dir>/config.json.

Examples:
    python eval.py runs/astrm_hoi4d_v3 --split test            # best epoch
    python eval.py runs/astrm_hoi4d_v3 --split test --checkpoint 37
    python eval.py runs/astrm_hoi4d_v3 --split test --nms_window 12
"""
import argparse
import os
import re

import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                                    # method dir: model/, dataset/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root: config, common
from model.astrm_model import ASTRMModel
from dataset.frame import ActionSpotVideoDataset
from common.spot_dataset import load_classes
from common.io import load_json
from train_astrm import evaluate, NMS_WINDOW, EVAL_TOLERANCES


def find_best_epoch(save_dir):
    loss_path = os.path.join(save_dir, 'loss.json')
    if os.path.exists(loss_path):
        rows = [r for r in load_json(loss_path) if r.get('val_mAP') is not None]
        if rows:
            best = max(rows, key=lambda r: r['val_mAP'])
            return best['epoch'], best['val_mAP']
    eps = [int(re.findall(r'checkpoint_(\d+)\.pt', f)[0])
           for f in os.listdir(save_dir) if re.match(r'checkpoint_\d+\.pt$', f)]
    return (max(eps), None) if eps else (None, None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('save_dir', help='Training run dir (has config.json + ckpts)')
    p.add_argument('--split', default='test', choices=['val', 'test', 'train'])
    p.add_argument('--checkpoint', default='best',
                   help="'best' (by val mAP), an epoch int, or a .pt path")
    p.add_argument('--frame_dir', default=None, help='Override frame dir')
    p.add_argument('--nms_window', type=int, default=None)
    args = p.parse_args()

    cfg = argparse.Namespace(
        **load_json(os.path.join(args.save_dir, 'config.json')))
    frame_dir = args.frame_dir or cfg.frame_dir
    nms_window = args.nms_window if args.nms_window is not None \
        else getattr(cfg, 'nms_window', NMS_WINDOW)

    classes = load_classes(os.path.join('data', cfg.dataset, 'class.txt'))
    num_classes = len(classes) + 1
    print('Dataset: {} | classes: {} | mAP@{}'.format(
        cfg.dataset, classes, EVAL_TOLERANCES))

    amp = {'bf16': torch.bfloat16, 'fp16': torch.float16,
           'fp32': torch.float32}[cfg.amp_dtype]
    astrm_kwargs = {}
    if getattr(cfg, 'astrm_temporal_kernel', None) is not None:
        astrm_kwargs['temporal_kernel'] = cfg.astrm_temporal_kernel
    model = ASTRMModel(
        num_classes, cfg.feature_arch, clip_len=cfg.clip_len,
        cls_loss=cfg.cls_loss, fg_weight=cfg.fg_weight,
        use_soft_ic=bool(cfg.use_soft_ic), lambda_sic=cfg.lambda_sic,
        amp_dtype=amp, astrm_kwargs=astrm_kwargs)

    if args.checkpoint == 'best':
        epoch, val_mAP = find_best_epoch(args.save_dir)
        assert epoch is not None, 'No checkpoints in {}'.format(args.save_dir)
        ckpt = os.path.join(
            args.save_dir, 'checkpoint_{:03d}.pt'.format(epoch))
        print('Best epoch {} (val mAP {})'.format(
            epoch, None if val_mAP is None else round(val_mAP * 100, 2)))
    elif args.checkpoint.endswith('.pt'):
        ckpt = args.checkpoint
    else:
        ckpt = os.path.join(
            args.save_dir, 'checkpoint_{:03d}.pt'.format(int(args.checkpoint)))
    print('Loading', ckpt)
    model.load(torch.load(ckpt))

    data = ActionSpotVideoDataset(
        classes, os.path.join('data', cfg.dataset, args.split + '.json'),
        frame_dir, 'rgb', cfg.clip_len, overlap_len=cfg.clip_len // 2,
        crop_dim=cfg.crop_dim)
    data.print_info()

    # evaluate() prints mAP@{0,1,2,4} for none / hard / soft in one pass.
    evaluate(model, data, args.split.upper(), classes, save_pred=None,
             nms_window=nms_window, nms_mode='soft')


if __name__ == '__main__':
    main()
