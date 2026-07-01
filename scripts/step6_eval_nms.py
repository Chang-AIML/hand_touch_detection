#!/usr/bin/env python3
"""Stage 6: test-set mAP@{0,1,2,4} for each spot_head head (TSP-MSTCN /
TSP-ASFormer) under three post-processings of the per-frame predictions:
  - none : raw dense high-recall predictions (no suppression)
  - NMS  : hard non-maximum suppression (peak picking) within +-window frames
  - SNMS : Gaussian Soft-NMS (decay near-duplicates by exp(-d^2/sigma))

It reuses the predictions already saved by methods/spot_head/train_head.py
(<save_dir>/pred-test.<best_epoch>.recall.json.gz) -- no retraining needed.
For each model it prints a headline table (chosen window/sigma) plus a sweep
over window and sigma so the full picture is visible, and writes everything to
<SPOT_HEAD_OUT>/nms_eval_results.{json,txt}.
"""
import os, sys, glob, argparse
import numpy as np
from tabulate import tabulate

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import config                                                       # noqa: E402
from common.io import load_json, load_gz_json                        # noqa: E402
from common.eval import (non_maximum_supression,                     # noqa: E402
                       soft_non_maximum_supression)
from common.score import (parse_ground_truth, get_predictions,       # noqa: E402
                        compute_average_precision)

TOLS = [0, 1, 2, 4]


def eval_map(truth, pred):
    """Return {'mAP':[per-tol], '<label>':[per-tol]} as percentages."""
    truth_by_label = parse_ground_truth(truth)
    labels = sorted(truth_by_label)
    per_label = {lab: [] for lab in labels}
    mAP = []
    for tol in TOLS:
        aps = []
        for lab in labels:
            ap = compute_average_precision(
                get_predictions(pred, label=lab), truth_by_label[lab], tolerance=tol)
            per_label[lab].append(ap * 100)
            aps.append(ap)
        mAP.append(float(np.mean(aps)) * 100)
    out = {'mAP': mAP}
    out.update(per_label)
    return out


def find_recall_pred(save_dir):
    cands = sorted(glob.glob(os.path.join(save_dir, 'pred-test.*.recall.json.gz')))
    if not cands:
        return None
    # if several, take the highest epoch index
    def ep(p):
        try:
            return int(os.path.basename(p).split('.')[1])
        except Exception:
            return -1
    return max(cands, key=ep)


def methods(pred, nms_window, snms_window):
    return [
        ('none', pred),
        (f'NMS(w={nms_window})', non_maximum_supression(pred, nms_window)),
        (f'SoftNMS(w={snms_window})', soft_non_maximum_supression(pred, snms_window)),
    ]


def fmt_table(rows):
    return tabulate(rows,
                    headers=['method', 'mAP@0', 'mAP@1', 'mAP@2', 'mAP@4', 'Avg'],
                    floatfmt='0.2f')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['mstcn', 'asformer'])
    ap.add_argument('--label_dir', default=config.LABEL_DIR)
    ap.add_argument('--spot_head_out', default=config.SPOT_HEAD_OUT)
    ap.add_argument('--nms_window', type=int, default=1,
                    help='hard-NMS window in frames (hice/E2E-Spot default 1)')
    ap.add_argument('--snms_window', type=int, default=4,
                    help='Soft-NMS window in frames (hice parabolic kernel)')
    ap.add_argument('--nms_sweep', type=int, nargs='+', default=[1, 2, 3, 4])
    ap.add_argument('--snms_sweep', type=int, nargs='+', default=[2, 4, 6, 8])
    args = ap.parse_args()

    truth = load_json(os.path.join(args.label_dir, 'test.json'))
    report = {'tolerances': TOLS, 'nms_window': args.nms_window,
              'snms_window': args.snms_window, 'models': {}}
    lines = []

    def out(s=''):
        print(s); lines.append(s)

    for arch in args.models:
        save_dir = os.path.join(args.spot_head_out, arch)
        pf = find_recall_pred(save_dir)
        out('\n' + '=' * 72)
        out(f'MODEL: TSP-{arch.upper()}   (test set, {len(truth)} videos)')
        out('=' * 72)
        if pf is None:
            out(f'  [skip] no pred-test.*.recall.json.gz under {save_dir} '
                f'(run methods/spot_head/train_head.py -m {arch} first)')
            continue
        out(f'  predictions: {os.path.relpath(pf)}')
        pred = load_gz_json(pf)

        # ---- headline: none / NMS / SoftNMS at chosen params ----
        head_rows, march = [], {}
        for name, pp in methods(pred, args.nms_window, args.snms_window):
            r = eval_map(truth, pp)
            head_rows.append([name, *r['mAP'], float(np.mean(r['mAP']))])
            march[name] = r
        out('\n-- mAP@tol (headline) --')
        out(fmt_table(head_rows))

        # ---- per-class AP for the chosen methods ----
        out('\n-- per-class AP --')
        labels = sorted([k for k in next(iter(march.values())) if k != 'mAP'])
        pc_rows = []
        for name in march:
            for lab in labels:
                pc_rows.append([name, lab, *march[name][lab]])
        out(tabulate(pc_rows, headers=['method', 'class', 'AP@0', 'AP@1', 'AP@2', 'AP@4'],
                     floatfmt='0.2f'))

        # ---- sweeps ----
        out('\n-- NMS window sweep --')
        sw_rows = [['none', *eval_map(truth, pred)['mAP'],
                    float(np.mean(eval_map(truth, pred)['mAP']))]]
        for w in args.nms_sweep:
            r = eval_map(truth, non_maximum_supression(pred, w))
            sw_rows.append([f'NMS(w={w})', *r['mAP'], float(np.mean(r['mAP']))])
        out(fmt_table(sw_rows))

        out('\n-- Soft-NMS window sweep --')
        ss_rows = []
        for w in args.snms_sweep:
            r = eval_map(truth, soft_non_maximum_supression(pred, w))
            ss_rows.append([f'SoftNMS(w={w})', *r['mAP'], float(np.mean(r['mAP']))])
        out(fmt_table(ss_rows))

        report['models'][arch] = {'pred_file': pf, 'headline': march}

    os.makedirs(args.spot_head_out, exist_ok=True)
    from common.io import store_json
    store_json(os.path.join(args.spot_head_out, 'nms_eval_results.json'), report, pretty=True)
    with open(os.path.join(args.spot_head_out, 'nms_eval_results.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    out(f'\nsaved -> {os.path.join(args.spot_head_out, "nms_eval_results.json")} (+ .txt)')


if __name__ == '__main__':
    main()
