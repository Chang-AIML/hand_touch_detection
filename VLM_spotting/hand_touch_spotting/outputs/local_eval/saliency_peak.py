"""JOB A — saliency-peak test: does the connector make EVENT frames DISTINGUISHABLE from NON-EVENT frames?

Stage-2 of the query→anchor→saliency→frame model REQUIRES that, within a window, the motion token at the
event frame stands out from non-event frames. Since FrameCompress RMS-matches every token to the SAME norm,
any such "standing out" must be DIRECTIONAL. We test separability of event vs non-event motion-token directions.

  event frame  = within ±TOL of a GT event ; non-event = the rest of the window
  metrics: (1) cos(mean_event_dir, mean_nonevent_dir)  ~1 => same direction => UNIFORM key (no peak)
           (2) AUC separating event vs non-event by projection on (mu_event - mu_nonevent)  0.5=chance
           (3) per-window: is an event frame the token FARTHEST (in that axis) from the window mean? (outlier rate)
Connector-only (s750); no LLM. Read: high AUC & low cos => saliency PEAKS (stage-2 viable);
AUC~0.5 & cos~1 => uniform key => explains @0 collapse (stage-2 can't localize the exact frame).
"""
import os, sys, json, numpy as np, torch
ROOT = os.environ.get("HTS_ROOT", "/data/code/hand_touch_spotting")
sys.path.insert(0, ROOT)
from dpc.vjepa import OnlineVJEPA
from models.frame_compress import FrameCompress
from dpc import paths

TOL = 2
CONN = os.environ.get("CONN", "/data/runs/mixed/p2_slide_a70_conn_s750.pt")
ANN = paths.ANN_DIR
LOCAL = os.environ.get("LOCAL_FRAMES", "")
if LOCAL:
    from dpc.frame_source import DirFrameSource; fs = DirFrameSource(LOCAL)
else:
    from dpc.frame_source import TarFrameSource; fs = TarFrameSource(paths.DATA_ROOT, paths.INDEX)

dev = "cuda"
fc = FrameCompress(768, 8, 4096, n_heads=4).to(dev, torch.float32)
fc.load_state_dict(torch.load(CONN, map_location="cpu", weights_only=False)["fc"], strict=True)
# RMS-match is a per-token scalar -> does NOT change direction; skip it (cosine analysis is norm-invariant).
vj = OnlineVJEPA(device=dev, dtype=torch.float16, compute_dtype=torch.bfloat16)

ev_tok, ne_tok = [], []          # event / non-event motion-token directions (unit)
win_outlier = []                 # per-window: is an event frame the max on the discriminant axis?
per_window = []                  # (event_dirs, nonevent_dirs) to score the axis per window
KEYS = ["finegym", "tennis", "touchmoment", "fs_perf"]
NW = int(os.environ.get("NW", "8"))   # windows per dataset
for k in KEYS:
    recs = json.load(open(f"{ANN}/{k}/val.json"))[:NW]
    for r in recs:
        vid = r["video_id"]; N = int(r["num_frames"])
        fr = fs.load_window(vid, 0, min(N, 600)); T = int(fr.shape[0])
        if T < 16:
            continue
        with torch.no_grad():
            motion = fc(vj.extract_grid(fr), None, T).float().cpu().numpy()   # (T,4096)
        motion = motion / (np.linalg.norm(motion, axis=1, keepdims=True) + 1e-8)
        evf = sorted({f for e in r.get("events", []) for f in range(int(e["frame"]) - TOL, int(e["frame"]) + TOL + 1) if 0 <= f < T})
        evset = set(evf); nef = [f for f in range(T) if f not in evset]
        if not evf or not nef:
            continue
        E = motion[evf]; NE = motion[nef]
        ev_tok.append(E); ne_tok.append(NE)
        per_window.append((E, NE))

E = np.concatenate(ev_tok); NE = np.concatenate(ne_tok)
muE = E.mean(0); muE /= np.linalg.norm(muE)
muNE = NE.mean(0); muNE /= np.linalg.norm(muNE)
axis = muE - muNE; axis /= (np.linalg.norm(axis) + 1e-8)
sE = E @ axis; sNE = NE @ axis
# AUC = P(score_event > score_nonevent)
auc = np.mean(sE[:, None] > sNE[None, :])
# per-window outlier: on this global axis, is an event frame among the top-1 of the whole window?
out = 0; tot = 0
for Ew, NEw in per_window:
    allw = np.concatenate([Ew, NEw]); s = allw @ axis
    ne_lab = [0] * len(Ew) + [1] * len(NEw)   # 0=event
    top = np.argmax(s)
    out += (ne_lab[top] == 0); tot += 1

print("\n================ SALIENCY-PEAK TEST (connector s750, direction-only) ================")
print(f"windows used: {len(per_window)} | event tokens: {len(E)} | non-event tokens: {len(NE)}")
print(f"(1) cos(mean_event_dir, mean_nonevent_dir) = {float(muE @ muNE):.3f}   (~1 => UNIFORM key / no peak)")
print(f"(2) AUC(event vs non-event, discriminant axis) = {auc:.3f}   (0.5 => not separable)")
print(f"(3) per-window: an EVENT frame is the axis-argmax in {out}/{tot} = {100*out/max(1,tot):.0f}% of windows (chance ~ frac event frames)")
print("READ: high AUC(>0.7) & low cos & high outlier% => saliency PEAKS (stage-2 viable).")
print("      AUC~0.5 & cos~1 => UNIFORM 'here's-an-event' key => stage-2 can't pick the exact frame => explains @0.")
