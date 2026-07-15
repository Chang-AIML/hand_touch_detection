"""Convert HOI4D-v3 labels -> exp_plan §3.2 annotation schema with questions.

Source: repos/astrm/data/hoi4d_v3/{train,val,test}.json  (list of videos with
events [{label, frame, comment}]). Output: data/annotations/{split}.json where
each video gets, per event, a natural-language `question` and an ordinal `type`
tag, plus a flat `samples` list (one (video, question, gt_frame) per event) that
Phase 1 consumes directly.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from data.questions import question_for  # noqa: E402

SRC_DIR = os.environ.get(
    "HOI4D_LABELS",
    "/home/chang_noroot/data2/huyanh/Workspace/repos/astrm/data/hoi4d_v3")
OUT_DIR = os.path.join(ROOT, "data", "annotations")


def convert_split(split: str):
    src = os.path.join(SRC_DIR, f"{split}.json")
    videos = json.load(open(src))
    out_videos, samples = [], []
    for v in videos:
        events = sorted(v["events"], key=lambda e: e["frame"])
        # ordinal per type
        counts = {t: sum(1 for e in events if e["label"] == t) for t in ("touch", "untouch")}
        seen = {"touch": 0, "untouch": 0}
        out_events = []
        for e in events:
            t = e["label"]
            q = question_for(t, seen[t], counts[t])
            seen[t] += 1
            oe = {"type": t, "frame": int(e["frame"]), "question": q}
            out_events.append(oe)
            samples.append({
                "video_id": v["video"], "fps": v["fps"],
                "num_frames": v["num_frames"], "type": t,
                "frame": int(e["frame"]), "question": q,
            })
        out_videos.append({
            "video_id": v["video"], "fps": v["fps"],
            "num_frames": v["num_frames"],
            "width": v.get("width"), "height": v.get("height"),
            "events": out_events,
        })
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(out_videos, open(os.path.join(OUT_DIR, f"{split}.json"), "w"), indent=1)
    json.dump(samples, open(os.path.join(OUT_DIR, f"{split}_samples.json"), "w"), indent=1)
    print(f"[{split}] {len(out_videos)} videos, {len(samples)} event-samples "
          f"-> {OUT_DIR}/{split}.json")
    return out_videos, samples


if __name__ == "__main__":
    for sp in ("train", "val", "test"):
        convert_split(sp)
    # sanity: show one sample
    ex = json.load(open(os.path.join(OUT_DIR, "val_samples.json")))[0]
    print("example sample:", ex)
