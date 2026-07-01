#!/usr/bin/env python3
"""Self-contained spot_head trainer: precise-spotting head (MS-TCN / ASFormer /
GRU / GCN) on top of the per-frame TSP features. Adapted from spot/baseline.py
but reads class.txt + train/val/test.json + feature dir from config (no external
spot repo needed; the head/eval code is vendored under methods/spot_head/ + common/).

Usage:
  python methods/spot_head/train_head.py -m mstcn
  python methods/spot_head/train_head.py -m asformer --num_epochs 50 --clip_len 100
"""
import os, sys, copy, random, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ChainedScheduler, LinearLR, CosineAnnealingLR
from tabulate import tabulate

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)     # vendored model/ dataset/ util/
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))         # touch_tsp/ for config
import config                                                     # noqa: E402
from model.feature_heads import GRU, TCN, GCN, ASFormer           # noqa: E402
from dataset.feature_dataset import FeatureDataset                # noqa: E402
from common.io import store_json, store_gz_json                     # noqa: E402
from common.eval import ForegroundF1, ErrorStat                     # noqa: E402
from common.spot_dataset import load_classes                             # noqa: E402
from common.score import compute_mAPs                               # noqa: E402

EPOCH_NUM_FRAMES = 1000000


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-m', '--model_arch', required=True,
                   choices=['gru', 'tcn', 'mstcn', 'gcn', 'asformer'])
    p.add_argument('--clip_len', type=int, default=config.SPOT_HEAD_CLIP_LEN)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_epochs', type=int, default=config.SPOT_HEAD_NUM_EPOCHS)
    p.add_argument('--warm_up_epochs', type=int, default=3)
    p.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    p.add_argument('--feat_dir', default=config.FEATURES_OUT)
    p.add_argument('--label_dir', default=config.LABEL_DIR)
    p.add_argument('--save_dir', default=None,
                   help='default: <SPOT_HEAD_OUT>/<arch>')
    p.add_argument('--feat_dims', type=int, nargs=2)
    p.add_argument('--dilate_len', type=int, default=0)
    return p.parse_args()


def evaluate(model, dataset, classes, save_pred=None, clip_len=None):
    classes_inv = {v: k for k, v in classes.items()}
    err, f1 = ErrorStat(), ForegroundF1()
    pred_events, pred_events_high_recall = [], []
    for video in dataset.videos:
        feat, label, pad_len = dataset.get(video)
        assert feat.shape[0] == label.shape[0] + 2 * pad_len
        if clip_len:
            scores = np.zeros((feat.shape[0], len(classes) + 1))
            support = np.zeros(feat.shape[0], dtype=np.int32)
            for i in range(0, max(1, feat.shape[0] - clip_len // 2 + 1), clip_len // 2):
                tmp = model.predict(feat[i:i + clip_len, :])[1]
                if i + tmp.shape[0] > feat.shape[0]:
                    tmp = tmp[:feat.shape[0] - i, :]
                scores[i:i + tmp.shape[0], :] += tmp
                support[i:i + tmp.shape[0]] += 1
            assert np.min(support) > 0
            scores /= support[:, None]
            pred = np.argmax(scores, axis=1)
        else:
            pred, scores = model.predict(feat)
        if pad_len > 0:
            pred, scores = pred[pad_len:-pad_len], scores[pad_len:-pad_len]
        err.update(label, pred)
        events, events_hr = [], []
        for i in range(len(pred)):
            f1.update(label[i], pred[i])
            if pred[i] != 0:
                events.append({'label': classes_inv[pred[i]], 'frame': i,
                               'score': scores[i, pred[i]].item()})
            for j in classes_inv:
                if scores[i, j] >= 0.01:
                    events_hr.append({'label': classes_inv[j], 'frame': i,
                                      'score': scores[i, j].item()})
        pred_events.append({'video': video, 'events': events})
        pred_events_high_recall.append({'video': video, 'events': events_hr})

    print('Error (frame-level): {:0.2f}\n'.format(err.get() * 100))
    rows = [['any', f1.get(None) * 100, *f1.tp_fp_fn(None)]]
    for c in sorted(classes):
        rows.append([c, f1.get(classes[c]) * 100, *f1.tp_fp_fn(classes[c])])
    print(tabulate(rows, headers=['Exact frame', 'F1', 'TP', 'FP', 'FN'], floatfmt='0.2f'))
    print()
    mAPs, _ = compute_mAPs(dataset._labels, pred_events_high_recall)
    print()
    if save_pred is not None:
        store_json(save_pred + '.json', pred_events)
        store_gz_json(save_pred + '.recall.json.gz', pred_events_high_recall)
    return np.mean(mAPs[1:])


def main(args):
    if args.save_dir is None:
        args.save_dir = os.path.join(config.SPOT_HEAD_OUT, args.model_arch)
    os.makedirs(args.save_dir, exist_ok=True)

    classes = load_classes(os.path.join(args.label_dir, 'class.txt'))
    dataset_len = EPOCH_NUM_FRAMES // args.clip_len
    mk = lambda split, dl: FeatureDataset(
        classes, os.path.join(args.label_dir, f'{split}.json'),
        args.feat_dir, args.clip_len, dl, feat_dims=args.feat_dims,
        dilate_len=args.dilate_len)
    train_data = mk('train', dataset_len);  train_data.print_info()
    val_data   = mk('val', dataset_len // 2); val_data.print_info()
    print('Feature dim:', train_data.feature_dim)

    global epoch; epoch = 0
    wif = lambda x: random.seed(x + epoch * 10)
    train_loader = DataLoader(train_data, shuffle=False, batch_size=args.batch_size, worker_init_fn=wif)
    val_loader   = DataLoader(val_data, shuffle=False, batch_size=args.batch_size, worker_init_fn=wif)

    nc = len(classes) + 1
    fd = train_data.feature_dim
    eval_clip = False
    if args.model_arch == 'gru':       model = GRU(fd, nc)
    elif args.model_arch == 'tcn':     model = TCN(fd, nc)
    elif args.model_arch == 'mstcn':   model = TCN(fd, nc, num_stages=3)
    elif args.model_arch == 'gcn':     model = GCN(fd, nc)
    elif args.model_arch == 'asformer':
        model = ASFormer(fd, nc); eval_clip = True
        print('ASFormer requires clip eval (learned position embedding)')

    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})
    steps = len(train_loader)
    cos = args.num_epochs - args.warm_up_epochs
    lr_sched = ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warm_up_epochs * steps),
        CosineAnnealingLR(optimizer, steps * cos)])

    losses, best_mAP, best_epoch, best_sd = [], 0, None, None
    for epoch in range(args.num_epochs):
        tl = model.epoch(train_loader, optimizer, scaler, lr_scheduler=lr_sched)
        vl = model.epoch(val_loader)
        print('[Epoch {}] Train loss: {:0.3f} Val loss: {:0.3f}'.format(epoch, tl, vl))
        losses.append({'train': tl, 'val': vl})
        store_json(os.path.join(args.save_dir, 'loss.json'), losses)
        print('=== VAL (epoch {}) ==='.format(epoch))
        mAP = evaluate(model, val_data, classes,
                       os.path.join(args.save_dir, f'pred-val.{epoch}'),
                       args.clip_len if eval_clip else None)
        if mAP > best_mAP:
            best_mAP, best_epoch = mAP, epoch
            best_sd = copy.deepcopy(model.state_dict())
            torch.save(best_sd, os.path.join(args.save_dir, 'best_epoch.pt'))
            print('New best epoch!')
    print('Best epoch: {} (val avg mAP {:0.2f})\n'.format(best_epoch, best_mAP))

    test_json = os.path.join(args.label_dir, 'test.json')
    if best_sd is not None and os.path.exists(test_json):
        model.load(best_sd)
        print('=== TEST (best epoch {}) ==='.format(best_epoch))
        test_data = FeatureDataset(classes, test_json, args.feat_dir, args.clip_len, 1,
                                   feat_dims=args.feat_dims)
        evaluate(model, test_data, classes,
                 os.path.join(args.save_dir, f'pred-test.{best_epoch}'),
                 args.clip_len if eval_clip else None)


if __name__ == '__main__':
    main(get_args())
