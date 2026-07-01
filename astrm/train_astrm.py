#!/usr/bin/env python3
""" Training for the ASTRM precise event-spotting reproduction.

ASTRM = E2E-Spot framework with: RegNetY+ASTRM backbone, Bi-GRU temporal block,
BCE classification loss + Soft-IC loss, ASAM optimizer, mixup (alpha=0.1).
Evaluated with mAP @ tolerance delta = 0, 1, 2 frames.
"""

import os
import argparse
import random
import numpy as np
import torch
torch.backends.cudnn.benchmark = True
from torch.optim.lr_scheduler import (
    SequentialLR, LinearLR, CosineAnnealingLR)
from torch.utils.data import DataLoader

from model.astrm_model import ASTRMModel
from model.asam import ASAM
from dataset.frame import ActionSpotDataset, ActionSpotVideoDataset
from util.eval import (process_frame_predictions, non_maximum_supression,
                       soft_non_maximum_supression)
from util.io import load_json, store_json, store_gz_json, clear_files
from util.dataset import DATASETS, load_classes
from util.score import compute_mAPs

EPOCH_NUM_FRAMES = 500000
BASE_NUM_WORKERS = 4
INFERENCE_BATCH_SIZE = 4
EVAL_TOLERANCES = [0, 1, 2, 4]       # delta, in frames (15 fps HOI4D)
NMS_WINDOW = 9                       # default NMS window (frames) for sparse mAP


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('dataset', choices=DATASETS)
    p.add_argument('frame_dir', help='Path to extracted frames')
    p.add_argument('-m', '--feature_arch', default='rny002_astrm',
                   choices=['rny002_astrm', 'rny008_astrm', 'rny002', 'rny008'])
    p.add_argument('--clip_len', type=int, default=128)
    p.add_argument('--crop_dim', type=int, default=224)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('-ag', '--acc_grad_iter', type=int, default=1)
    p.add_argument('--warm_up_epochs', type=int, default=3)
    p.add_argument('--num_epochs', type=int, default=50)
    p.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    p.add_argument('-s', '--save_dir', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--start_val_epoch', type=int)
    p.add_argument('--dilate_len', type=int, default=0)
    p.add_argument('--mixup', type=int, default=1)
    p.add_argument('--mixup_alpha', type=float, default=0.1)
    p.add_argument('--fg_upsample', type=float)
    p.add_argument('--fg_weight', type=float, default=1)
    # ASTRM-specific knobs
    p.add_argument('--cls_loss', choices=['ce', 'bce'], default='bce')
    p.add_argument('--use_asam', type=int, default=1)
    p.add_argument('--asam_rho', type=float, default=2.0)
    p.add_argument('--use_soft_ic', type=int, default=1)
    p.add_argument('--lambda_sic', type=float, default=0.001)
    p.add_argument('--astrm_temporal_kernel', type=int, default=7,
                   help='Global-temporal dynamic-kernel length K. Default 7, '
                        'back-inferred from the paper 60.25 GFLOPs (K=128 '
                        'is ~1.42x too heavy).')
    p.add_argument('--amp_dtype', choices=['bf16', 'fp16', 'fp32'],
                   default='bf16')
    # NMS is optional post-processing; we always report no-NMS / hard / soft and
    # select the best epoch on --nms_mode.
    p.add_argument('--nms_window', type=int, default=NMS_WINDOW,
                   help='NMS window in frames (hard & soft).')
    p.add_argument('--nms_mode', choices=['none', 'hard', 'soft'],
                   default='soft',
                   help='Which post-processing the model-selection mAP uses.')
    p.add_argument('-j', '--num_workers', type=int)
    p.add_argument('-mgpu', '--gpu_parallel', action='store_true')
    p.add_argument('--epoch_num_frames', type=int, default=EPOCH_NUM_FRAMES)
    return p.parse_args()


def evaluate(model, dataset, split, classes, save_pred, calc_stats=True,
             nms_window=NMS_WINDOW, nms_mode='soft'):
    pred_dict = {}
    for video, video_len, _ in dataset.videos:
        pred_dict[video] = (
            np.zeros((video_len, len(classes) + 1), np.float32),
            np.zeros(video_len, np.int32))

    batch_size = 1 if dataset.augment else INFERENCE_BATCH_SIZE
    eval_workers = min(8, BASE_NUM_WORKERS)
    for clip in DataLoader(dataset, num_workers=eval_workers,
                           pin_memory=True, batch_size=batch_size):
        if batch_size > 1:
            _, batch_pred_scores = model.predict(clip['frame'])
            for i in range(clip['frame'].shape[0]):
                video = clip['video'][i]
                scores, support = pred_dict[video]
                pred_scores = batch_pred_scores[i]
                start = clip['start'][i].item()
                if start < 0:
                    pred_scores = pred_scores[-start:, :]
                    start = 0
                end = start + pred_scores.shape[0]
                if end >= scores.shape[0]:
                    end = scores.shape[0]
                    pred_scores = pred_scores[:end - start, :]
                scores[start:end, :] += pred_scores
                support[start:end] += 1
        else:
            scores, support = pred_dict[clip['video'][0]]
            start = clip['start'][0].item()
            _, pred_scores = model.predict(clip['frame'][0])
            if start < 0:
                pred_scores = pred_scores[:, -start:, :]
                start = 0
            end = start + pred_scores.shape[1]
            if end >= scores.shape[0]:
                end = scores.shape[0]
                pred_scores = pred_scores[:, :end - start, :]
            scores[start:end, :] += np.sum(pred_scores, axis=0)
            support[start:end] += pred_scores.shape[0]

    err, f1, pred_events, pred_events_high_recall, pred_scores = \
        process_frame_predictions(dataset, classes, pred_dict)

    avg_mAP = None
    if calc_stats:
        print('=== Results on {} ==='.format(split))
        print('Error (frame-level): {:0.2f}'.format(err.get() * 100))
        # NMS is optional post-processing -- report all three variants.
        mAPs = {}
        print('--- mAP@{0,1,2} without NMS (dense metric) ---')
        mAPs['none'], _ = compute_mAPs(
            dataset.labels, pred_events_high_recall, tolerances=EVAL_TOLERANCES)
        print('--- mAP@{0,1,2} with hard-NMS (window=%d) ---' % nms_window)
        mAPs['hard'], _ = compute_mAPs(
            dataset.labels,
            non_maximum_supression(pred_events_high_recall, nms_window),
            tolerances=EVAL_TOLERANCES)
        print('--- mAP@{0,1,2} with soft-NMS (window=%d) ---' % nms_window)
        mAPs['soft'], _ = compute_mAPs(
            dataset.labels,
            soft_non_maximum_supression(pred_events_high_recall, nms_window),
            tolerances=EVAL_TOLERANCES)
        # Model selection uses the chosen post-processing (default soft-NMS).
        avg_mAP = float(np.mean(mAPs[nms_mode]))
        print('>> selection metric = {}-NMS, avg mAP = {:0.2f}'.format(
            nms_mode, avg_mAP * 100))

    if save_pred is not None:
        store_json(save_pred + '.json', pred_events)
        store_gz_json(save_pred + '.recall.json.gz', pred_events_high_recall)
    return avg_mAP


def get_datasets(args, classes):
    dataset_len = args.epoch_num_frames // args.clip_len
    kw = dict(crop_dim=args.crop_dim, dilate_len=args.dilate_len,
              mixup=bool(args.mixup), mixup_alpha=args.mixup_alpha)
    if args.fg_upsample is not None:
        kw['fg_upsample'] = args.fg_upsample
    data_dir = os.path.join('data', args.dataset)
    train_data = ActionSpotDataset(
        classes, os.path.join(data_dir, 'train.json'), args.frame_dir, 'rgb',
        args.clip_len, dataset_len, is_eval=False, **kw)
    train_data.print_info()
    val_data_frames = ActionSpotVideoDataset(
        classes, os.path.join(data_dir, 'val.json'), args.frame_dir, 'rgb',
        args.clip_len, crop_dim=args.crop_dim,
        overlap_len=args.clip_len // 2)
    return train_data, val_data_frames


def get_lr_scheduler(args, optimizer, steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    warmup_steps = args.warm_up_epochs * steps_per_epoch
    cosine = CosineAnnealingLR(optimizer, steps_per_epoch * cosine_epochs)
    if warmup_steps <= 0:
        print('Cosine only ({} epochs)'.format(cosine_epochs))
        return cosine
    print('Linear warmup ({}) -> cosine ({}) [sequential]'.format(
        args.warm_up_epochs, cosine_epochs))
    # SequentialLR: warmup first, THEN cosine (not both multiplied together).
    return SequentialLR(optimizer, [
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=warmup_steps),
        cosine], milestones=[warmup_steps])


def main(args):
    if args.num_workers is not None:
        global BASE_NUM_WORKERS
        BASE_NUM_WORKERS = args.num_workers
    if args.start_val_epoch is None:
        args.start_val_epoch = max(0, args.num_epochs - 20)

    classes = load_classes(os.path.join('data', args.dataset, 'class.txt'))
    num_classes = len(classes) + 1
    print('Classes:', classes, '| num_classes (w/ bg):', num_classes)

    train_data, val_data_frames = get_datasets(args, classes)

    loader_bs = args.batch_size // args.acc_grad_iter

    def worker_init_fn(wid):
        # torch gives each worker a fresh seed per epoch; mirror it into the
        # python `random` module that the dataset's sampling/augmentation uses,
        # so clips & augmentations vary across epochs.
        random.seed(torch.initial_seed() % (2 ** 31))

    # Memory guard: RAM held by the loader ~= num_workers * prefetch_factor *
    # (batch_size clips). With large batch this explodes, so keep prefetch low.
    n_workers = min(os.cpu_count(), BASE_NUM_WORKERS * 2)
    train_loader = DataLoader(
        train_data, shuffle=False, batch_size=loader_bs, pin_memory=True,
        num_workers=n_workers,
        prefetch_factor=1 if n_workers > 0 else None,
        worker_init_fn=worker_init_fn)

    amp = {'bf16': torch.bfloat16, 'fp16': torch.float16,
           'fp32': torch.float32}[args.amp_dtype]
    astrm_kwargs = {}
    if args.astrm_temporal_kernel is not None:
        astrm_kwargs['temporal_kernel'] = args.astrm_temporal_kernel
    model = ASTRMModel(
        num_classes, args.feature_arch, clip_len=args.clip_len,
        cls_loss=args.cls_loss, fg_weight=args.fg_weight,
        use_soft_ic=bool(args.use_soft_ic), lambda_sic=args.lambda_sic,
        amp_dtype=amp, multi_gpu=args.gpu_parallel,
        astrm_kwargs=astrm_kwargs)
    optimizer, _ = model.get_optimizer({'lr': args.learning_rate})
    minimizer = ASAM(optimizer, model._core, rho=args.asam_rho) \
        if args.use_asam else None

    steps_per_epoch = len(train_loader) // args.acc_grad_iter
    lr_scheduler = get_lr_scheduler(args, optimizer, steps_per_epoch)

    os.makedirs(args.save_dir, exist_ok=True)
    store_json(os.path.join(args.save_dir, 'config.json'), vars(args),
               pretty=True)

    losses, best_epoch, best_mAP = [], None, -1
    start_epoch = 0
    if args.resume:
        start_epoch, losses, best_epoch, best_mAP = load_from_save(
            args, model, optimizer, lr_scheduler)
        start_epoch += 1

    for epoch in range(start_epoch, args.num_epochs):
        train_loss = model.epoch(
            train_loader, optimizer, minimizer=minimizer,
            lr_scheduler=lr_scheduler, acc_grad_iter=args.acc_grad_iter)
        print('[Epoch {}] Train loss: {:0.5f}'.format(epoch, train_loss))

        # Save the trained weights BEFORE evaluating, so a crash during the
        # (worker-heavy) validation never loses an epoch of training.
        torch.save(model.state_dict(),
                   os.path.join(args.save_dir,
                                'checkpoint_{:03d}.pt'.format(epoch)))
        clear_files(args.save_dir, r'optim_\d+\.pt')
        torch.save({'optimizer_state_dict': optimizer.state_dict(),
                    'lr_state_dict': lr_scheduler.state_dict()},
                   os.path.join(args.save_dir,
                                'optim_{:03d}.pt'.format(epoch)))

        val_mAP = 0
        if epoch >= args.start_val_epoch:
            pred_file = os.path.join(args.save_dir, 'pred-val.{}'.format(epoch))
            try:
                val_mAP = evaluate(
                    model, val_data_frames, 'VAL', classes, pred_file,
                    nms_window=args.nms_window, nms_mode=args.nms_mode)
            except Exception as e:
                print('!! VAL eval failed at epoch {}: {}'.format(epoch, e))
                val_mAP = 0
            if val_mAP > best_mAP:
                best_mAP, best_epoch = val_mAP, epoch
                print('New best epoch! val mAP {:0.2f}'.format(val_mAP * 100))

        losses.append({'epoch': epoch, 'train': train_loss,
                       'val_mAP': val_mAP})
        store_json(os.path.join(args.save_dir, 'loss.json'), losses, pretty=True)

    print('Best epoch: {} (val mAP {:0.2f})'.format(
        best_epoch, (best_mAP or 0) * 100))

    if best_epoch is not None:
        model.load(torch.load(os.path.join(
            args.save_dir, 'checkpoint_{:03d}.pt'.format(best_epoch))))
        test_path = os.path.join('data', args.dataset, 'test.json')
        if os.path.exists(test_path):
            test_data = ActionSpotVideoDataset(
                classes, test_path, args.frame_dir, 'rgb', args.clip_len,
                overlap_len=args.clip_len // 2, crop_dim=args.crop_dim)
            test_data.print_info()
            evaluate(model, test_data, 'TEST', classes,
                     os.path.join(args.save_dir,
                                  'pred-test.{}'.format(best_epoch)),
                     nms_window=args.nms_window, nms_mode=args.nms_mode)


def get_last_epoch(save_dir):
    mx = -1
    for f in os.listdir(save_dir):
        if f.startswith('optim_'):
            mx = max(mx, int(os.path.splitext(f)[0].split('optim_')[1]))
    return mx


def load_from_save(args, model, optimizer, lr_scheduler):
    epoch = get_last_epoch(args.save_dir)
    print('Resuming from epoch {}'.format(epoch))
    model.load(torch.load(os.path.join(
        args.save_dir, 'checkpoint_{:03d}.pt'.format(epoch))))
    opt = torch.load(os.path.join(
        args.save_dir, 'optim_{:03d}.pt'.format(epoch)))
    optimizer.load_state_dict(opt['optimizer_state_dict'])
    lr_scheduler.load_state_dict(opt['lr_state_dict'])
    losses = load_json(os.path.join(args.save_dir, 'loss.json'))
    best = max(losses, key=lambda x: x['val_mAP'])
    return epoch, losses, best['epoch'], best['val_mAP']


if __name__ == '__main__':
    main(get_args())
