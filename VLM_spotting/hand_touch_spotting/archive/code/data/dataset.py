"""Phase 1 dataset: one sample = (video, single question, single GT frame).

Loads the pre-extracted per-frame V-JEPA interleave features (N,768). Embedding
construction (adaptor + question + [LOC] + assemble) happens in the train step
because it needs the model on-GPU with grad, so the collate_fn just returns the
list of light per-sample dicts.
"""
from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


class Phase1Dataset(Dataset):
    def __init__(self, split: str, feat_dir: str, types=("touch", "untouch"),
                 require_feats: bool = True, crop_secs: int = 0, p_neg: float = 0.0):
        """crop_secs>0 -> random temporal crop of that many 1-fps seconds, aligned
        to fps boundaries so the event lands at a RANDOM offset (breaks the absolute
        event-position prior) while keeping ViT-anchor caching valid. 0 -> full video.
        p_neg -> fraction of crops that DELIBERATELY exclude the event (gt=-1); the
        loss makes s(t) flat there so the model rejects non-event windows (needed for
        full-video sliding inference)."""
        samples = json.load(open(os.path.join(ROOT, "data", "annotations", f"{split}_samples.json")))
        self.feat_dir = feat_dir
        self.crop_secs = crop_secs
        self.p_neg = p_neg
        self.samples = []
        for s in samples:
            if s["type"] not in types:
                continue
            if require_feats and not os.path.exists(os.path.join(feat_dir, s["video_id"] + ".npy")):
                continue
            self.samples.append(s)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        feats = np.load(os.path.join(self.feat_dir, s["video_id"] + ".npy")).astype(np.float32)
        N_full = feats.shape[0]
        fps = int(s["fps"])
        gt = int(s["frame"])
        num_secs_full = N_full // fps
        start_sec, num_secs = 0, num_secs_full
        is_neg = 0
        if self.crop_secs and num_secs_full > self.crop_secs:
            gt_sec = min(gt // fps, num_secs_full - 1)
            num_secs = self.crop_secs
            # candidate negative windows (exclude the event's second entirely)
            neg_starts = [st for st in range(0, num_secs_full - num_secs + 1)
                          if not (st <= gt_sec < st + num_secs)]
            if self.p_neg > 0 and neg_starts and random.random() < self.p_neg:
                start_sec = random.choice(neg_starts)
                is_neg = 1
            else:
                lo = max(0, gt_sec - num_secs + 1)
                hi = min(gt_sec, num_secs_full - num_secs)
                start_sec = random.randint(lo, hi)
            start = start_sec * fps
            W = num_secs * fps
            feats = feats[start:start + W]
            gt = -1 if is_neg else (gt - start)
        return {
            "feats": torch.from_numpy(np.ascontiguousarray(feats)),   # (N,768)
            "question": s["question"],
            "gt": int(gt),
            "fps": fps,
            "type": s["type"],
            "video_id": s["video_id"],
            "num_frames": feats.shape[0],
            "full_num_frames": N_full,
            "anchor_start_sec": start_sec,
            "anchor_num_secs": num_secs,
            "is_neg": is_neg,
        }


def collate_list(batch):
    return batch


from data.questions import GENERIC_Q, sample_question  # noqa: E402,F401 (re-export)


def video_category(video_id):
    import re
    m = re.match(r"H\d+_C(\d+)_", video_id)
    return int(m.group(1)) if m else -1


class DetectionDataset(Dataset):
    """Multi-event detection: one item = (video, event-type) with a GENERIC (no-ordinal)
    query. Target = ALL frames of that type in the (cropped) window. Random temporal
    crop centred on a random same-type event; with prob p_neg an event-ABSENT window
    (empty target). This matches the data (2-5 touches/video) and the baseline's
    per-frame multi-event detection protocol."""

    def __init__(self, split: str, feat_dir: str, types=("touch", "untouch"),
                 crop_secs: int = 0, p_neg: float = 0.2, question_phase: str = "train",
                 categories=None, exclude_categories=None):
        vids = json.load(open(os.path.join(ROOT, "data", "annotations", f"{split}.json")))
        self.feat_dir = feat_dir
        self.crop_secs = crop_secs
        self.p_neg = p_neg
        self.question_phase = question_phase          # 'train' (diverse) | 'test' (held-out phrasings)
        self.items = []          # (video_id, type, [sorted same-type frames], fps, N)
        for v in vids:
            if not os.path.exists(os.path.join(feat_dir, v["video_id"] + ".npy")):
                continue
            c = video_category(v["video_id"])
            if categories is not None and c not in categories:
                continue
            if exclude_categories is not None and c in exclude_categories:
                continue
            for typ in types:
                fr = sorted(e["frame"] for e in v["events"] if e["type"] == typ)
                if fr:
                    self.items.append((v["video_id"], typ, fr, v["fps"], v["num_frames"]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        vid, typ, frames, fps, N_full = self.items[i]
        feats = np.load(os.path.join(self.feat_dir, vid + ".npy")).astype(np.float32)
        N_full = feats.shape[0]
        num_secs_full = N_full // fps
        start_sec, num_secs = 0, num_secs_full
        ev_local = list(frames)
        if self.crop_secs and num_secs_full > self.crop_secs:
            num_secs = self.crop_secs
            W = num_secs * fps
            neg_starts = [st for st in range(0, num_secs_full - num_secs + 1)
                          if not any(st * fps <= f < st * fps + W for f in frames)]
            if self.p_neg > 0 and neg_starts and random.random() < self.p_neg:
                start_sec = random.choice(neg_starts)            # event-absent window
            else:
                anchor = random.choice(frames); a_sec = min(anchor // fps, num_secs_full - 1)
                lo = max(0, a_sec - num_secs + 1); hi = min(a_sec, num_secs_full - num_secs)
                start_sec = random.randint(lo, hi)
            start = start_sec * fps
            feats = feats[start:start + W]
            ev_local = [f - start for f in frames if start <= f < start + W]
        return {
            "feats": torch.from_numpy(np.ascontiguousarray(feats)),
            "question": sample_question(typ, self.question_phase),   # diverse paraphrase
            "event_frames": ev_local,          # ALL same-type events in window (local coords)
            "gt": ev_local[0] if ev_local else -1,   # for meta compatibility
            "fps": fps, "type": typ, "video_id": vid,
            "num_frames": feats.shape[0], "full_num_frames": N_full,
            "anchor_start_sec": start_sec, "anchor_num_secs": num_secs,
            "is_neg": int(len(ev_local) == 0),
        }
