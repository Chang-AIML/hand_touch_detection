#!/usr/bin/env python3
"""Score saved test predictions under three post-processings and report mAP@[0,1,2,4]:
  - without NMS   : the dense high-recall per-frame predictions (what train_head reports)
  - with NMS      : hard temporal NMS (E2E-Spot util.eval.non_maximum_supression), window=1 default
  - with Soft-NMS : Gaussian temporal soft-NMS (decay nearby same-class scores, keep all)

Reads each mode's pred-test.<best>.recall.json.gz (saved by methods/spot_head/train_head.py) and the
test.json labels. Produces one combined table across modes x methods.

Usage:
  python methods/spot_head/eval_nms.py --modes interleave even odd stack \
      --spot-head-dir outputs/spot_head --prefix vjepa_mstcn_ --split test
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))            # repo root for config
import config                                                         # noqa: E402
from common.io import load_json, load_gz_json                           # noqa: E402
from common.eval import (non_maximum_supression,                        # noqa: E402
                        soft_non_maximum_supression)                    # hice util/eval.py
from common.score import (compute_mAPs, parse_ground_truth,             # noqa: E402
                        get_predictions, compute_average_precision)

TOLS = [0, 1, 2, 4]


def per_class_aps(truth, pred):
    """Return {label: [AP@0, AP@1, AP@2, AP@4]} in % (touch / untouch separately)."""
    tbl = parse_ground_truth(truth)
    return {label: [compute_average_precision(get_predictions(pred, label=label), tfl,
                                              tolerance=t) * 100 for t in TOLS]
            for label, tfl in tbl.items()}


def maps_quiet(truth, pred):
    """compute_mAPs but swallow its internal table print; return list of mAP per tolerance."""
    with contextlib.redirect_stdout(io.StringIO()):
        mAPs, _ = compute_mAPs(truth, pred, tolerances=TOLS)
    return [m * 100 for m in mAPs]


def find_recall_pred(mode_dir, split):
    g = sorted(glob.glob(os.path.join(mode_dir, f'pred-{split}.*.recall.json.gz')))
    return g[-1] if g else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modes', nargs='+', default=['interleave', 'even', 'odd', 'stack'])
    ap.add_argument('--spot-head-dir', default=config.SPOT_HEAD_OUT)
    ap.add_argument('--prefix', default='vjepa_mstcn_')
    ap.add_argument('--split', default='test')
    ap.add_argument('--label-dir', default=config.LABEL_DIR)
    ap.add_argument('--nms-window', type=int, default=1, help='hice hard-NMS window (E2E-Spot default 1)')
    ap.add_argument('--snms-window', type=int, default=4, help='hice soft-NMS window (frames)')
    ap.add_argument('--per-class', action='store_true', help='also print touch/untouch AP separately')
    ap.add_argument('--tolerances', nargs='+', type=int, default=[0, 1, 2, 4],
                    help='tolerances to report; the Avg column is their mean (e.g. 0 1 2 for @0/1/2 mAP)')
    args = ap.parse_args()
    global TOLS
    TOLS = args.tolerances

    truth = load_json(os.path.join(args.label_dir, f'{args.split}.json'))

    methods = ['without NMS', f'NMS (w={args.nms_window})',
               f'Soft-NMS (w={args.snms_window})']

    def variants(pred_hr):
        return [pred_hr,
                non_maximum_supression(pred_hr, args.nms_window),
                soft_non_maximum_supression(pred_hr, args.snms_window)]

    results = {}                                  # (mode, method) -> [mAP@0,1,2,4]
    cls_results = {}                              # (mode, method, label) -> [AP@0,1,2,4]
    for mode in args.modes:
        mdir = os.path.join(args.spot_head_dir, f'{args.prefix}{mode}')
        rp = find_recall_pred(mdir, args.split)
        if rp is None:
            print(f'[skip] {mode}: no pred-{args.split}.*.recall.json.gz in {mdir}', flush=True)
            continue
        pred_hr = load_gz_json(rp)
        for meth, pv in zip(methods, variants(pred_hr)):
            results[(mode, meth)] = maps_quiet(truth, pv)
            if args.per_class:
                for label, aps in per_class_aps(truth, pv).items():
                    cls_results[(mode, meth, label)] = aps
        print(f'[done] {mode}  <- {os.path.basename(rp)}', flush=True)

    def table(title, getrow):
        hdr = f'{"mode":<11} {"method":<22} ' + ' '.join(f'@{t:<6}' for t in TOLS) + 'Avg'
        print('\n' + '=' * len(hdr)); print(title); print('=' * len(hdr)); print(hdr); print('-' * len(hdr))
        for mode in args.modes:
            any_row = False
            for meth in methods:
                v = getrow(mode, meth)
                if v is None:
                    continue
                any_row = True
                print(f'{mode:<11} {meth:<22} ' + ' '.join(f'{x:<7.2f}' for x in v) + f'{sum(v)/len(v):.2f}')
            if any_row:
                print('-' * len(hdr))

    table(f'V-JEPA MS-TCN — {args.split} set  mAP (%) [touch & untouch averaged]',
          lambda m, me: results.get((m, me)))
    if args.per_class:
        for label in ['touch', 'untouch']:
            table(f'V-JEPA MS-TCN — {args.split} set  AP (%) — class: {label}',
                  lambda m, me, _l=label: cls_results.get((m, me, _l)))


if __name__ == '__main__':
    main()
