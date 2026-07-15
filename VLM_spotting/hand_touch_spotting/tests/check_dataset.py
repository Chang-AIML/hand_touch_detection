"""Tier-2 dataset determinism (no GPU): build a training WindowedSpottingDataset on a fixed
2-dataset subset with the real p2 negative params, and snapshot index length + neg_stats +
first-30 index tuples + temperature_epoch_order(0)[:30] + 5 question samples.
  tests/check_dataset.py capture  -> tests/golden/dataset_golden.json  (run on CURRENT code)
  tests/check_dataset.py          -> compare a fresh build against that snapshot
This guards the training-path RNG (type1/2/3 negatives, neg_cap, temperature_epoch_order, _question)
that the end-to-end golden eval does NOT exercise."""
import sys, os, json, tempfile
sys.path.insert(0, ".")
from dpc.windowed_dataset import WindowedSpottingDataset
from dpc.frame_source import DirFrameSource

MODE = sys.argv[1] if len(sys.argv) > 1 else "check"
ANN = os.environ["AS_ANN_DIR"]
KEYS = ["tennis", "fs_perf"]                       # 2 datasets -> exercises the type-3 cross-dataset negs
recs = []
for k in KEYS:
    recs += json.load(open(f"{ANN}/{k}/train.json"))
tmp = tempfile.mktemp(suffix=".json"); json.dump(recs, open(tmp, "w"))

ds = WindowedSpottingDataset(tmp, DirFrameSource(os.environ["AS_DATA_ROOT"]),
                             window_frames=600, stride=600, question_phase="train",
                             negative_rate=0.15, cross_neg_rate=0.15, type2_rate=1.5,
                             neg_cap=0.40, neg_cap_by_ds={"finegym": 0.05}, temp_alpha=0.70,
                             jitter=30, seed=0)
questions = [ds._question(ds.clips[e[0]], e[2], e[0], prefix=e[3]) for e in ds.index[:5]]
snap = {"len": len(ds.index), "neg_stats": ds.neg_stats,
        "index_head": [list(e) for e in ds.index[:30]],
        "epoch0_head": ds.temperature_epoch_order(0)[:30],
        "questions": questions}
path = "tests/golden/dataset_golden.json"
if MODE == "capture":
    json.dump(snap, open(path, "w"), indent=1)
    print(f"captured dataset golden: {snap['len']} samples, neg_stats={snap['neg_stats']}")
else:
    gold = json.load(open(path))
    if json.dumps(snap, sort_keys=True) == json.dumps(gold, sort_keys=True):
        print(f"DATASET PASS: determinism identical ({snap['len']} samples)")
    else:
        for k in gold:
            if json.dumps(snap[k], sort_keys=True) != json.dumps(gold[k], sort_keys=True):
                print(f"  MISMATCH {k}:\n   new={str(snap[k])[:160]}\n   old={str(gold[k])[:160]}")
        sys.exit(1)
