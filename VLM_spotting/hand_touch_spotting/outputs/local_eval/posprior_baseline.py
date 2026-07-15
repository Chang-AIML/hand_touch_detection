"""Position-prior baseline for finediving OOD, at every tolerance (CPU-only, no model).

Reconstructs the eval's OOD scoring but replaces the model prediction with the DUMB baseline:
for each (window,type), predict the per-type MEDIAN within-window event position (score 1.0).
If a broken/garbage connector (e.g. MIX-3, ID@2=0.3) scores near this baseline on finediving, then
finediving's OOD numbers are a small-window position-prior artifact, not genuine transfer.
"""
import os, sys, statistics as st
sys.path.insert(0, "/home/chang/Project/VLM_spotting/hand_touch_spotting")
from dpc import paths
from dpc.eval_vendor import eval_nms as en
from dpc.windowed_dataset import WindowedSpottingDataset
from dpc.frame_source import DirFrameSource

en.TOLS = [0, 1, 2, 4, 8, 16]
ANN = os.environ["AS_ANN_DIR"]
# build the finediving OOD eval set exactly like train_mixed (val split, balance = full 1018)
import json, glob
combined = f"{ANN}/finediving/val.json"
ds = WindowedSpottingDataset(combined, DirFrameSource(os.environ["AS_DATA_ROOT"]),
                             window_frames=600, stride=600, question_phase="test",
                             balance_per_dataset=1018, seed=0)

# per-window (type, window-local targets) straight from the index (no frame loading)
wins = []
for (ci, ws, typ, qp) in ds.index:
    W = ds.W
    tgt = [f - ws for f in ds._by_type[ci].get(typ, []) if ws <= f < ws + W]
    vid = f"{ds.clips[ci]['video_id']}#w{ws}"
    wins.append((vid, typ, tgt))

# per-type median within-window position (the "prior")
bypos = {}
for _, typ, tgt in wins:
    bypos.setdefault(typ, []).extend(tgt)
median_pos = {t: int(round(st.median(v))) for t, v in bypos.items() if v}
print(f"[data] {len(wins)} finediving windows, {len(median_pos)} types; median positions: {median_pos}")

truth, pred = [], []                              # scorer wants per-video {'video','events':[...]}
for vid, typ, tgt in wins:
    truth.append({"video": vid, "events": [{"label": typ, "frame": int(f)} for f in tgt]})
    ev = [{"label": typ, "frame": median_pos[typ], "score": 1.0}] if typ in median_pos else []
    pred.append({"video": vid, "events": ev})

m = en.maps_quiet(truth, pred)
ng = sum(len(x["events"]) for x in truth); nd = sum(len(x["events"]) for x in pred)
print(f"\n[POSITION-PRIOR BASELINE  finediving]  mAP@{tuple(en.TOLS)} = {[round(x,2) for x in m]}  ndet={nd} ngt={ng}")
print(f"[MIX-3 OOD (QF1200+out750, ID collapsed 0.31)]                 = [0.44, 4.17, 9.32, 26.2, 43.83, 59.24]")
print(f"[MIX-4 OOD (QF750+out1200)]                                    = [0.14, 1.15, 3.83, 16.77, 36.87, 48.52]")
print(f"[s750 OOD (matched, ID=39.9)]                                  = [0.24, 3.01, 7.68, 26.45, 48.17, 65.81]")
print(f"[s1200 OOD (matched, ID=50.2)]                                 = [0.20, 1.73, 4.43, 15.44, 31.36, 38.62]")
print("READ: if MIX-3 OOD ~ baseline -> its 'OOD' is position prior (broken connector). If s750 >> baseline -> genuine transfer.")
