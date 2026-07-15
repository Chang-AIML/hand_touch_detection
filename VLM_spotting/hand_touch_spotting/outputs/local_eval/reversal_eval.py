"""Temporal-reversal probe (s750): does the connector encode the ARROW OF TIME (real motion
direction) or only SALIENCY (direction-agnostic change)?

finegym is the only domain with paired start/end of the SAME element (e.g. BB_dismounts_start at
f_s, BB_dismounts_end at f_e, f_s < f_e = takeoff then landing). We run each clip TWICE — forward and
with frames reversed (frame p -> N-1-p) — with the START query and the END query, and ask what the
reversed START query locks onto.

  reversed video: at the original LANDING frame the reversed motion looks like a TAKEOFF, and vice versa.
  * DIRECTION encoded  -> reversed START-query fires at rev_e = N-1-f_e (the reversed end) -> SWAP
  * SALIENCY only      -> reversed START-query fires at rev_s = N-1-f_s (mirrored start pos) -> NO swap
  * POSITION prior     -> reversed START-query fires at a fixed early frame regardless of content

We report, per query, whether the reversed prediction is closer to the SWAPPED target (direction) or the
MIRRORED target (saliency/position), and the absolute-position bias, so the three hypotheses separate.

Runs on the local refactored code. Usage: python reversal_eval.py [conn.pt] [N_windows]
"""
import os, sys, json, random
sys.path.insert(0, "/home/chang/Project/VLM_spotting/hand_touch_spotting")
import numpy as np
import torch

CONN = sys.argv[1] if len(sys.argv) > 1 else "outputs/local_eval/conn_s750.pt"
NW = int(sys.argv[2]) if len(sys.argv) > 2 else 60
ANN = os.environ["AS_ANN_DIR"]; ROOT = os.environ["AS_DATA_ROOT"]
GAP = 40                                   # min f_e - f_s so start/end are distinguishable

from dpc.frame_source import DirFrameSource
from dpc.vjepa import OnlineVJEPA
from dpc.questions import question_for
from models.wrapper import QwenWrapper
from models.frame_compress import FrameCompress
from models.localizer import Localizer

# ---- pick paired (start,end) windows from finegym val ----
recs = json.load(open(f"{ANN}/finegym/val.json"))
pairs = []                                 # (rec, base, f_s, f_e)
for r in recs:
    N = int(r["num_frames"]); by = {}
    for e in r.get("events", []):
        by.setdefault(e["type"], []).append(int(e["frame"]))
    for st in [t for t in by if t.endswith("_start")]:
        base = st[:-6]; en = base + "_end"
        if en in by and len(by[st]) == 1 and len(by[en]) == 1:   # unambiguous single pair
            fs, fe = by[st][0], by[en][0]
            if 0 <= fs < fe < min(N, 600) and fe - fs >= GAP:
                pairs.append((r, base, fs, fe))
random.Random(0).shuffle(pairs)
pairs = pairs[:NW]
print(f"[data] {len(pairs)} paired start/end finegym windows (gap>={GAP})")

# ---- model (s750) ----
dev = "cuda"
os.environ.setdefault("VJEPA_REPO", "/home/chang/Project/vlm_deps/vjepa2")
W = QwenWrapper(model_id=os.environ["QWEN_PATH"], device=dev, dtype=torch.bfloat16)
fc = FrameCompress(768, 8, W.d_llm, n_heads=4).to(dev, torch.float32)
fc.load_state_dict(torch.load(CONN, map_location="cpu", weights_only=False)["fc"], strict=True)
fc.set_target_rms_from(W.embed_tokens.weight)
loz = Localizer(W, fc, use_anchor=True, anchor_stride=5, anchor_max_side=252, fps=15,
                max_frames=600, grad_checkpoint=False)
vjepa = OnlineVJEPA(device=dev, dtype=torch.float16, compute_dtype=torch.bfloat16)
fsrc = DirFrameSource(ROOT)


def q_of(base, boundary):                   # the fixed eval ('general') query for start/end of this element
    return question_for("finegym", f"{base}_{boundary}", phase="test")


def predict(frames, N, fps, question):
    grid = vjepa.extract_grid(frames)
    s = {"frames": frames, "grid": grid, "num_frames": N, "fps": fps, "question": question,
         "anchor_start_sec": 0, "anchor_num_secs": int(np.ceil(N / fps)),
         "video_id": "rev", "dataset": "finegym", "event_frames": [], "type": "x"}
    frames_pred, _scores = loz.predict_multievent_batch([s], max_new_tokens=64)[0]
    fr = [f for f in frames_pred if 0 <= f < N]
    return (fr[0] if fr else -1), fr             # primary (first) predicted frame


rows = []
for i, (r, base, fs, fe) in enumerate(pairs):
    vid = r["video_id"]; N = int(r["num_frames"]); fps = float(r.get("fps") or 15)
    fps = int(round(fps))
    fwd = fsrc.load_window(vid, 0, min(N, 600))
    T = int(fwd.shape[0]); N = T
    rev = np.ascontiguousarray(fwd[::-1])
    qs, qe = q_of(base, "start"), q_of(base, "end")
    ps_f, _ = predict(fwd, N, fps, qs); pe_f, _ = predict(fwd, N, fps, qe)     # forward
    ps_r, _ = predict(rev, N, fps, qs); pe_r, _ = predict(rev, N, fps, qe)     # reversed
    rev_s, rev_e = N - 1 - fs, N - 1 - fe                                       # mapped targets
    rows.append(dict(base=base, N=N, fs=fs, fe=fe, ps_f=ps_f, pe_f=pe_f,
                     ps_r=ps_r, pe_r=pe_r, rev_s=rev_s, rev_e=rev_e))
    if i % 10 == 0:
        print(f"  {i+1}/{len(pairs)} {base:22s} fs={fs} fe={fe} | fwd s->{ps_f} e->{pe_f} | rev s->{ps_r} e->{pe_r} (rev_s={rev_s} rev_e={rev_e})")

# ---- analysis ----
def near(p, a, b):                          # is p closer to a (target A) than b (target B)? None if no pred
    return None if p < 0 else (abs(p - a) <= abs(p - b))

fwd_ok = [r for r in rows if r["ps_f"] >= 0 and r["pe_f"] >= 0]
fwd_start_before_end = np.mean([r["ps_f"] < r["pe_f"] for r in fwd_ok]) if fwd_ok else float("nan")
fwd_start_err = np.mean([abs(r["ps_f"] - r["fs"]) for r in fwd_ok]) if fwd_ok else float("nan")
fwd_end_err = np.mean([abs(r["pe_f"] - r["fe"]) for r in fwd_ok]) if fwd_ok else float("nan")

# reversed START query: swap target = rev_e (direction), mirror target = rev_s (saliency)
swap_start = [near(r["ps_r"], r["rev_e"], r["rev_s"]) for r in rows]
swap_end = [near(r["pe_r"], r["rev_s"], r["rev_e"]) for r in rows]
swap_start = [x for x in swap_start if x is not None]
swap_end = [x for x in swap_end if x is not None]
# absolute-position bias: reversed start pred normalized position (0=clip start, 1=clip end)
rev_start_pos = np.mean([r["ps_r"] / max(1, r["N"] - 1) for r in rows if r["ps_r"] >= 0])
rev_end_pos = np.mean([r["pe_r"] / max(1, r["N"] - 1) for r in rows if r["pe_r"] >= 0])

print("\n================ RESULTS ================")
print(f"forward sanity: start localizes @{fwd_start_err:.1f}f err, end @{fwd_end_err:.1f}f err, "
      f"start<end in {fwd_start_before_end*100:.0f}% of windows  (N={len(fwd_ok)})")
print(f"REVERSED start-query -> SWAP(=near rev_e, direction) in {np.mean(swap_start)*100:.0f}%  (N={len(swap_start)})")
print(f"REVERSED end-query   -> SWAP(=near rev_s, direction) in {np.mean(swap_end)*100:.0f}%  (N={len(swap_end)})")
print(f"reversed abs-pos: start-query mean pos={rev_start_pos:.2f}, end-query mean pos={rev_end_pos:.2f} (0=clip start,1=end)")
print("READ: swap>>50% -> direction encoded (temporal). swap~=50% or abs-pos fixed by query -> saliency/position only.")
json.dump(rows, open("outputs/local_eval/reversal_rows.json", "w"), indent=1)
print("rows -> outputs/local_eval/reversal_rows.json")
