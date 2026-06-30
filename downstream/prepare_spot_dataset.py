#!/usr/bin/env python3
"""Register the HOI4D-touch spotting dataset into the spot repo so baseline.py
can train MS-TCN / ASFormer on the TSP features.

Does two idempotent things:
  1. create  <SPOT_REPO>/data/<DS_NAME>/{class.txt, train.json, val.json, test.json}
     (symlinks to our E2E-Spot label splits in config.LABEL_DIR)
  2. add  '<DS_NAME>'  to DATASETS in <SPOT_REPO>/util/dataset.py

Spotting classes = the event labels in the json (touch / untouch). The TSP
[T,512] features in config.FEATURES_OUT are the per-frame inputs.
"""
import os, sys, shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


def _link(src, dst):
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def main():
    spot, name = config.SPOT_REPO, config.DOWNSTREAM_DATASET
    assert os.path.isdir(spot), f'spot repo not found: {spot} (set TOUCH_SPOT_REPO)'

    # 1) data/<name>/
    ddir = os.path.join(spot, 'data', name)
    os.makedirs(ddir, exist_ok=True)
    # class.txt: prefer the one next to the labels, else write touch/untouch
    src_cls = os.path.join(config.LABEL_DIR, 'class.txt')
    if os.path.exists(src_cls):
        _link(src_cls, os.path.join(ddir, 'class.txt'))
    else:
        with open(os.path.join(ddir, 'class.txt'), 'w') as f:
            f.write('touch\nuntouch\n')
    for split in ['train', 'val', 'test']:
        src = os.path.join(config.LABEL_DIR, f'{split}.json')
        if os.path.exists(src):
            _link(src, os.path.join(ddir, f'{split}.json'))
    print(f'[1] dataset dir ready: {ddir}')
    print('    ', os.listdir(ddir))

    # 2) register in DATASETS
    du = os.path.join(spot, 'util', 'dataset.py')
    src = open(du).read()
    if f"'{name}'" in src.split('DATASETS = [')[1].split(']')[0]:
        print(f'[2] {name!r} already in DATASETS')
    else:
        src = src.replace('DATASETS = [', f"DATASETS = [\n    '{name}',", 1)
        shutil.copy(du, du + '.bak')
        open(du, 'w').write(src)
        print(f'[2] added {name!r} to DATASETS (backup: {du}.bak)')

    print(f'\nFeatures: {config.FEATURES_OUT}')
    print('Ready. Run downstream/train_downstream.sh')


if __name__ == '__main__':
    main()
