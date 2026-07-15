"""idx-decode dataset — SINGLE-EVENT clean windows for the 'LLM copies the frame
index' experiment.

One item = (video, one event of a type). We crop a window of `crop_secs` around
that event, CONSTRAINED so the window contains exactly ONE same-type event (the
target) — no ambiguity, so a correct answer is unambiguous and the copy/count
hypothesis can be tested cleanly. The question is a GENERIC per-type paraphrase
(no ordinal). Target = the event's LOCAL frame index in the window (0..W-1).

Returns light dicts; the embedding sequence (motion tokens + idx digits +
question + answer) is built on-GPU in the train step (models/idx_localizer.py).
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

from data.questions import sample_question, GENERIC_Q  # noqa: E402


def _video_category(video_id):
    import re
    m = re.match(r"H\d+_C(\d+)_", video_id)
    return int(m.group(1)) if m else -1


class IdxSingleEventDataset(Dataset):
    def __init__(self, split: str, feat_dir: str, types=("touch", "untouch"),
                 crop_secs: int = 6, question_phase: str = "train",
                 deterministic: bool = False, categories=None, exclude_categories=None):
        vids = json.load(open(os.path.join(ROOT, "data", "annotations", f"{split}.json")))
        self.feat_dir = feat_dir
        self.crop_secs = crop_secs
        self.question_phase = question_phase
        self.deterministic = deterministic          # eval: center the event, fixed crop
        self.items = []       # (video_id, type, target_frame, [same-type frames], fps, N)
        for v in vids:
            if not os.path.exists(os.path.join(feat_dir, v["video_id"] + ".npy")):
                continue
            c = _video_category(v["video_id"])
            if categories is not None and c not in categories:
                continue
            if exclude_categories is not None and c in exclude_categories:
                continue
            for typ in types:
                same = sorted(e["frame"] for e in v["events"] if e["type"] == typ)
                for fe in same:
                    self.items.append((v["video_id"], typ, fe, same, v["fps"], v["num_frames"]))

    def __len__(self):
        return len(self.items)

    def _pick_start_sec(self, fe, siblings, num_secs, num_secs_full, fps):
        """Choose an FPS-ALIGNED window start (in seconds) whose [start, start+W)
        frame span CONTAINS fe and EXCLUDES all same-type siblings. fps-aligned so
        the 1-fps ViT anchors slice cleanly. Returns start_sec or None."""
        W = num_secs * fps
        others = [f for f in siblings if f != fe]
        lo = max(0, -(-(fe - W + 1) // fps))          # ceil((fe-W+1)/fps)
        hi = min(fe // fps, num_secs_full - num_secs)
        valid = [ss for ss in range(lo, hi + 1)
                 if all(not (ss * fps <= o < ss * fps + W) for o in others)]
        if not valid:
            return None
        if self.deterministic:
            target = max(0, min(fe // fps - num_secs // 2, num_secs_full - num_secs))
            return min(valid, key=lambda s: abs(s - target))
        return random.choice(valid)

    def __getitem__(self, i):
        vid, typ, fe, siblings, fps, N_full = self.items[i]
        feats = np.load(os.path.join(self.feat_dir, vid + ".npy")).astype(np.float32)
        N_full = feats.shape[0]
        num_secs_full = N_full // fps
        num_secs = self.crop_secs
        if num_secs_full <= num_secs:
            start_sec, num_secs = 0, num_secs_full   # whole video (may be ambiguous)
        else:
            start_sec = self._pick_start_sec(fe, siblings, num_secs, num_secs_full, fps)
            if start_sec is None:                    # cannot isolate -> centered fallback (ambiguous)
                start_sec = max(0, min(fe // fps - num_secs // 2, num_secs_full - num_secs))
        start = start_sec * fps
        W = num_secs * fps
        feats = feats[start:start + W]
        gt_local = fe - start
        in_win = [f - start for f in siblings if start <= f < start + W]
        return {
            "feats": torch.from_numpy(np.ascontiguousarray(feats)),  # (W,768)
            "question": sample_question(typ, self.question_phase),
            "gt": int(gt_local),
            "n_same_in_window": len(in_win),         # 1 => clean single-event
            "fps": int(fps), "type": typ, "video_id": vid,
            "num_frames": feats.shape[0], "full_num_frames": N_full,
            "anchor_start_sec": start_sec, "anchor_num_secs": num_secs,
        }


class IdxMultiEventDataset(Dataset):
    """HONEST setting, comparable to the baseline: one item = (video, type) over the
    FULL video (300f). Target = the sorted list of ALL same-type event frames
    (GLOBAL coords). The LLM must output every event (or 'none'). No single-event
    isolation. Eval scores full-video multi-event mAP against all GT events."""

    def __init__(self, split: str, feat_dir: str, types=("touch", "untouch"),
                 question_phase: str = "train", categories=None, exclude_categories=None):
        vids = json.load(open(os.path.join(ROOT, "data", "annotations", f"{split}.json")))
        self.feat_dir = feat_dir
        self.question_phase = question_phase
        self.items = []       # (video_id, type, [sorted frames], fps, N)
        for v in vids:
            if not os.path.exists(os.path.join(feat_dir, v["video_id"] + ".npy")):
                continue
            c = _video_category(v["video_id"])
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
        frames = [f for f in frames if f < N_full]
        return {
            "feats": torch.from_numpy(np.ascontiguousarray(feats)),   # (N,768) full video
            "question": sample_question(typ, self.question_phase),
            "event_frames": frames,               # ALL same-type events, GLOBAL coords
            "gt": frames[0] if frames else -1,
            "fps": int(fps), "type": typ, "video_id": vid,
            "num_frames": N_full, "full_num_frames": N_full,
            "anchor_start_sec": 0, "anchor_num_secs": N_full // fps,
        }


def collate_list(batch):
    return batch
