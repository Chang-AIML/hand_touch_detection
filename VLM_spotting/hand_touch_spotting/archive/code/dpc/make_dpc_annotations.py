#!/usr/bin/env python3
"""Convert DPC Action_Spotting annotations -> training schema, generating NL questions.

Input  (loose DPC split json, unified schema):
  <root>/<ann_subpath>/<split>.json =
    [{"video","num_frames","num_events","events":[{"frame","label","comment"}],
      "fps","width","height"}]

Output (per registry key, per split, under <out>/<key>/):
  <split>.json         : [{"video_id","dataset","fps","num_frames","width","height",
                          "events":[{"type","frame","question"}]}]  (sorted by frame)
  <split>_samples.json : flattened, one row per event {video_id,dataset,fps,num_frames,type,frame,question}

video_id = "<video_prefix>/<video>"  -> matches the clip id in /data/Action_Spotting/index.sqlite.

Registry encodes the in-domain / out-of-domain split (finediving = OOD, test-only).
Datasets with NO extracted frames (fs_comp, native TouchMoment) are intentionally excluded.
"""
import os, json, argparse, random

try:
    from dpc.questions_multi import question_for
except Exception:
    from questions_multi import question_for

# (key, ann_subpath, video_id_prefix, in_domain)
REGISTRY = [
    ("touchmoment_hoi4d", "TouchMoment/Annotations/HOI4D", "TouchMoment", True),
    ("touchmoment_taco",  "TouchMoment/Annotations/TACO",  "TouchMoment", True),
    ("tennis",            "tennis",                        "tennis",      True),
    ("finegym",           "finegym",                       "finegym",     True),
    ("soccernet_ball",    "soccernet_ball",                "soccernet_ball", True),
    ("fs_perf",           "fs_comp",                       "fs_perf",     True),  # fs dataset now sources the fs_comp superset (371 clips); frames + video_id keep 'fs_perf' prefix
    ("finediving",        "finediving",                    "finediving",  False),  # OOD: test-only
]
SPLITS = ["train", "val", "test"]


def convert_split(entries, prefix, dataset_key, phase, rng):
    """entries: DPC list -> (nested_records, flat_samples)."""
    nested, flat = [], []
    for e in entries:
        vid = e["video"]
        video_id = f"{prefix}/{vid}"
        fps = e.get("fps"); nf = e.get("num_frames")
        # group events by type for ordinal/count
        by_type = {}
        for ev in e.get("events", []):
            by_type.setdefault(ev["label"], []).append(ev)
        out_events = []
        for label, evs in by_type.items():
            evs_sorted = sorted(evs, key=lambda x: x["frame"])
            cnt = len(evs_sorted)
            for i, ev in enumerate(evs_sorted):
                q = question_for(prefix, label, phase=phase, ordinal=i, count=cnt, rng=rng)
                rec = {"type": label, "frame": int(ev["frame"]), "question": q}
                out_events.append(rec)
                flat.append({"video_id": video_id, "dataset": dataset_key, "fps": fps,
                             "num_frames": nf, "type": label, "frame": int(ev["frame"]),
                             "question": q})
        out_events.sort(key=lambda x: x["frame"])
        nested.append({"video_id": video_id, "dataset": dataset_key, "fps": fps,
                       "num_frames": nf, "width": e.get("width"), "height": e.get("height"),
                       "events": out_events})
    return nested, flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/Action_Spotting", help="DPC dataset root")
    ap.add_argument("--out", required=True, help="output annotations dir")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    summary = []
    for key, sub, prefix, in_domain in REGISTRY:
        ann_dir = os.path.join(args.root, sub)
        if not os.path.isdir(ann_dir):
            print(f"[skip] {key}: no dir {ann_dir}"); continue
        out_dir = os.path.join(args.out, key); os.makedirs(out_dir, exist_ok=True)
        for split in SPLITS:
            src = os.path.join(ann_dir, f"{split}.json")
            if not os.path.exists(src):
                continue
            # OOD datasets contribute only val/test (never trained on)
            if not in_domain and split == "train":
                print(f"[ood ] {key}: skipping train split (out-of-domain, test-only)"); continue
            entries = json.load(open(src))
            # train-split gets 'train' phrasings; val/test get HELD-OUT 'test' phrasings
            phase = "train" if split == "train" else "test"
            nested, flat = convert_split(entries, prefix, key, phase, rng)
            json.dump(nested, open(os.path.join(out_dir, f"{split}.json"), "w"))
            json.dump(flat, open(os.path.join(out_dir, f"{split}_samples.json"), "w"))
            n_ev = sum(len(r["events"]) for r in nested)
            summary.append((key, split, len(nested), n_ev, in_domain))
            print(f"[ok  ] {key}/{split}: {len(nested)} clips, {n_ev} events -> {out_dir}")

    print("\n=== summary ===")
    print(f"{'key':20s} {'split':6s} {'clips':>7s} {'events':>9s}  domain")
    for key, split, nc, ne, ind in summary:
        print(f"{key:20s} {split:6s} {nc:7d} {ne:9d}  {'in' if ind else 'OOD'}")


if __name__ == "__main__":
    main()
