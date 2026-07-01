#!/usr/bin/env python3
"""HOI4D (touch/untouch event spotting) -> TSP DUAL-HEAD segment CSV.

TSP-native two-head setup (matches train.py / models.Model num_heads=2):
  - action head          : touch / untouch        (classified on Foreground clips only)
  - temporal-region head : Foreground / Background (GVF-fed)

The events are POINT events (a touch frame / an untouch frame). Following TSP's
definition (a clip is Foreground if its center lies inside an action region, and
gets that region's class; otherwise Background with no action label), we turn each
event into an event-centered Foreground window of radius FG_RADIUS frames, tagged
with the event's action class. Background = the complement.

With FG_RADIUS = CLIP_LEN//2 + 2, a CLIP_LEN clip sampled inside a window keeps the
event within +-2 of the clip center (well inside the half-clip tolerance of
E2E-Spot's (K+1)-VC). Background rows carry an EMPTY action-label -> the dataset
maps it to -1 so CrossEntropyLoss(ignore_index=-1) skips the action loss there.
Times in seconds (frame/fps)."""
import json, csv, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import config                       # noqa: E402
SRC = config.LABEL_DIR   # train.json/val.json/test.json live here
OUT = config.DATA_DIR    # write hoi4d_<split>_tsp.csv here
FG_RADIUS = config.CLIP_LEN // 2 + 2     # 8 frames for CLIP_LEN=12 -> 16-frame windows


def merge(intervals):
    """Union of [s,e) intervals (frames)."""
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def complement(union, n):
    """Background = [0,n) minus the union of Foreground windows."""
    out, prev = [], 0
    for s, e in union:
        if s > prev:
            out.append((prev, s))
        prev = max(prev, e)
    if prev < n:
        out.append((prev, n))
    return out


from collections import Counter
for split in ['train', 'val', 'test']:
    data = json.load(open(os.path.join(SRC, f'{split}.json')))
    rows = []
    fg_labels = Counter(); n_bg = 0     # per-action-class tally (dataset-agnostic)
    for v in data:
        name, fps, n = v['video'], float(v['fps']), v['num_frames']
        dur = round(n / fps, 4)
        fg = []                                              # (s_frame, e_frame, action)
        for ev in v['events']:
            f, lab = ev['frame'], ev['label']                # event class (touch/untouch, jump_takeoff, ...)
            s, e = max(0, f - FG_RADIUS), min(n, f + FG_RADIUS)
            if e > s:
                fg.append((s, e, lab))
        for s, e, lab in fg:                                 # Foreground rows (action = the event class)
            rows.append([name, fps, round(s / fps, 4), round(e / fps, 4), lab, 'Foreground', dur])
            fg_labels[lab] += 1
        for s, e in complement(merge([(s, e) for s, e, _ in fg]), n):   # Background rows (no action)
            if e - s < 1:
                continue
            rows.append([name, fps, round(s / fps, 4), round(e / fps, 4), '', 'Background', dur])
            n_bg += 1
    with open(os.path.join(OUT, f'{config.DATASET}_{split}_tsp.csv'), 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(['filename', 'fps', 't-start', 't-end',
                    'action-label', 'temporal-region-label', 'video-duration'])
        w.writerows(rows)
    print(f'{split:5s}: {len(data):4d} videos -> {len(rows):5d} rows '
          f'(FG {dict(fg_labels)}, BG {n_bg})')
