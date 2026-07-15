"""Action-head dataset: one item = (video, type, ONE GT event). Returns the full
V-JEPA feats + gt frame + a per-type question. The train step samples a jittered
center, slices the ±W window (pre-LLM adaptor), and regresses δ = gt - center."""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

from data.questions import sample_question, GENERIC_Q  # noqa: E402


class ActionHeadDataset(Dataset):
    def __init__(self, split: str, feat_dir: str, types=("touch", "untouch"),
                 question_phase: str = "train"):
        vids = json.load(open(os.path.join(ROOT, "data", "annotations", f"{split}.json")))
        self.feat_dir = feat_dir
        self.question_phase = question_phase
        self.items = []          # (video_id, type, gt_frame, fps, N)
        for v in vids:
            if not os.path.exists(os.path.join(feat_dir, v["video_id"] + ".npy")):
                continue
            for e in v["events"]:
                if e["type"] in types:
                    self.items.append((v["video_id"], e["type"], int(e["frame"]),
                                       int(v["fps"]), int(v["num_frames"])))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        vid, typ, gt, fps, N_full = self.items[i]
        feats = np.load(os.path.join(self.feat_dir, vid + ".npy")).astype(np.float32)
        return {
            "feats": torch.from_numpy(np.ascontiguousarray(feats)),   # (N,768)
            "type": typ, "gt": gt, "video_id": vid, "fps": fps,
            "num_frames": feats.shape[0],
            "question": sample_question(typ, self.question_phase),
        }


def collate_list(batch):
    return batch
