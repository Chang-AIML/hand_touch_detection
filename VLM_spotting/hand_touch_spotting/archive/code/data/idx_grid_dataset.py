"""Stage-1 dataset that serves the UNPOOLED even grid (N/2,576,768) for the language-
compression front-end, instead of the mean-pooled (N,768). Same items/questions/targets
as IdxMultiEventDataset. Grid loaded synchronously per item (no worker procs -> OOM-safe)."""
from __future__ import annotations

import json
import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
from data.questions import sample_question                        # noqa: E402
GRID = "/home/chang_noroot/data2/huyanh/Workspace/VLM_spotting/vjepa/grid_even"


def _cat(vid):
    m = re.match(r"H\d+_C(\d+)_", vid); return int(m.group(1)) if m else -1


class IdxMultiGridDataset(Dataset):
    def __init__(self, split, types=("touch", "untouch"), question_phase="train",
                 grid_dir=GRID, categories=None, exclude_categories=None):
        vids = json.load(open(os.path.join(ROOT, "data", "annotations", f"{split}.json")))
        self.grid_dir = grid_dir; self.question_phase = question_phase
        self.items = []
        for v in vids:
            vid = v["video_id"]
            if not os.path.exists(os.path.join(grid_dir, vid + ".npy")):
                continue
            c = _cat(vid)
            if categories is not None and c not in categories:
                continue
            if exclude_categories is not None and c in exclude_categories:
                continue
            for typ in types:
                fr = sorted(e["frame"] for e in v["events"] if e["type"] == typ)
                if fr:
                    self.items.append((vid, typ, fr, int(v["fps"]), int(v["num_frames"])))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        vid, typ, frames, fps, N = self.items[i]
        grid = np.load(os.path.join(self.grid_dir, vid + ".npy"))    # (N2,576,768) fp16
        N = min(N, 2 * grid.shape[0])                                # per-frame count from even grid
        frames = [f for f in frames if f < N]
        return {
            "grid": torch.from_numpy(grid),          # (N2,576,768) -> compressor interps to N
            "question": sample_question(typ, self.question_phase),
            "event_frames": frames, "gt": frames[0] if frames else -1,
            "fps": fps, "type": typ, "video_id": vid,
            "num_frames": N, "full_num_frames": N,
            "anchor_start_sec": 0, "anchor_num_secs": N // fps,
        }
