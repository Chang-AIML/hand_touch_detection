"""Windowed frame dataset for the 'idx-compress' spotting model.

Reads a clip's frames (via a FrameSource) + its events (from the converted nested
<split>.json produced by dpc/make_dpc_annotations.py), slices each clip into
fixed-length frame windows, and yields one training/eval sample per
(window x event-type-present-in-clip).

This is the raw-frame front of the winning path: it mirrors the sample semantics
of data/idx_dataset.py::IdxMultiEventDataset (per-(video,type) multi-event items,
'none' negatives, each clip's own fps, a per-type natural-language question) but
(a) operates on FRAME PIXELS instead of pre-extracted V-JEPA features, and
(b) crops the clip into fixed windows so long finegym/soccernet clips become many
    bounded samples instead of one giant sequence.
A SEPARATE module turns each sample's ``frames`` (uint8 T,H,W,3) into motion
tokens; nothing here loads torch models or V-JEPA — it is light and CPU-only.

Sample schema (one dict per __getitem__):
    video_id        : '<dataset>/<clipname>'            (matches FrameSource key)
    dataset         : registry key, e.g. 'touchmoment_hoi4d'
    question        : natural-language question for this event TYPE
    type            : raw event label, e.g. 'touch', 'FX_turns_start'
    target_frames   : window-LOCAL int indices of this type's events inside the
                      window; [] means a 'none' (event-type-absent) window
    win_start       : global frame index the window starts at
    num_win_frames  : number of frames actually loaded for this window (== T)
    fps             : the clip's OWN fps (int or float; never hardcoded)
    frames          : np.uint8 (T, H, W, 3) RGB pixels for this window
    num_frames      : full clip frame count (from the annotation)

Positives vs 'none' negatives:
  * For every (clip, type-present) we walk the windows. Windows that CONTAIN >=1
    event of that type are always emitted (target_frames non-empty).
  * Windows of that type with NO event inside are "event-type-absent" windows; a
    random ``negative_rate`` fraction of them is emitted as 'none' (target_frames
    == []). negative_rate=0.0 -> positives only; 1.0 -> every window x present-type.
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
from torch.utils.data import Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)                     # HTS repo root (has data/ package)
for _p in (_REPO, _HERE):                          # so data.questions / questions_multi import
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Natural-language question generator, same source used by make_dpc_annotations.py.
try:
    from dpc.questions_multi import question_for
except Exception:                                  # running with dpc/ on the path directly
    try:
        from questions_multi import question_for
    except Exception:
        question_for = None                        # fall back to baked annotation questions


class WindowedSpottingDataset(Dataset):
    def __init__(
        self,
        annotations_json: str,
        frame_source,
        window_frames: int = 320,
        stride: int | None = None,           # default = window_frames (non-overlapping)
        question_phase: str = "train",
        negative_rate: float = 0.0,          # fraction of event-type-absent windows -> 'none'
        subset: float | None = None,         # keep this fraction of clips PER dataset
        max_clips: int | None = None,        # ... or cap clips PER dataset (applied after subset)
        balance_per_dataset: int | None = None,  # cap (window,type) SAMPLES per dataset, spread
                                             # evenly across that dataset's clips (round-robin).
                                             # Fixes domain imbalance (finegym no longer dominates)
                                             # and starvation (few-clip sets like soccernet use all
                                             # their clips instead of being cut to 1 by clip-subset).
        jitter: int = 0,                     # train aug: random ± frame offset on the window start
                                             # (only for clips longer than the window) -> breaks the
                                             # fixed-grid event-POSITION prior. 0 = off (use for eval).
        cross_neg_rate: float = 0.0,         # per-window prob of adding a CROSS-DATASET hard negative:
                                             # ask a foreign dataset's action on this window -> NL
                                             # "no frame related" answer. Teaches "not present" + generalization.
        temp_alpha: float = 0.0,             # dataset-level temperature exponent for temperature_epoch_order:
                                             # per-epoch draws for dataset d ~ n_d^alpha (WITHOUT replacement -> full
                                             # coverage). 0.75 lets finegym get >=1 full pass while small sets get 1.5-2x.
        type2_rate: float = 0.0,             # per-window prob of an IN-DOMAIN hard negative: ask ANOTHER type from the
                                             # SAME dataset that is absent from this clip -> NL "no frame" (disambiguation).
        neg_cap: float = 0.40,               # cap negatives (type1/2/3) at this fraction of a dataset's samples; drop
                                             # random excess so the model doesn't over-predict "none" (fs is ~57% empty).
        neg_cap_by_ds: dict | None = None,   # per-dataset override of neg_cap (keyed by video_id prefix), e.g.
                                             # {'finegym': 0.20}: shrink the dominant set's pool so its POSITIVES get
                                             # fully covered at a lower alpha + less "none" bias from the big domain.
        seed: int = 0,
    ):
        self.jitter = int(jitter)
        self.temp_alpha = float(temp_alpha)
        self.type2_rate = float(type2_rate)
        self.neg_cap = float(neg_cap)
        self.neg_cap_by_ds = dict(neg_cap_by_ds or {})
        self.fs = frame_source
        self.W = int(window_frames)
        self.stride = int(stride) if stride else self.W
        assert self.W > 0 and self.stride > 0, "window_frames and stride must be > 0"
        self.question_phase = question_phase
        self.negative_rate = float(negative_rate)
        self.cross_neg_rate = float(cross_neg_rate)
        self.seed = int(seed)

        records = json.load(open(annotations_json))
        self.clips = self._subset_per_dataset(records, subset, max_clips)

        # Precompute per-clip: {type -> sorted [event frames]} + a baked-question pool
        # (used as fallback when questions_multi is unavailable).
        self._by_type = []      # list[dict[str, list[int]]]
        self._q_pool = []       # list[dict[str, list[str]]]
        for rec in self.clips:
            N = int(rec["num_frames"])
            bt, qp = {}, {}
            for ev in rec.get("events", []):
                f = int(ev["frame"])
                if 0 <= f < N:
                    bt.setdefault(ev["type"], []).append(f)
                    qp.setdefault(ev["type"], []).append(ev.get("question", ""))
            for t in bt:
                bt[t].sort()
            self._by_type.append(bt)
            self._q_pool.append(qp)

        # Per-clip dataset prefix (e.g. 'TouchMoment', 'tennis') + a GLOBAL registry of
        # (prefix, action_type) across all clips -> pool for cross-dataset hard negatives.
        self._clip_prefix = [rec["video_id"].split("/", 1)[0] for rec in self.clips]
        _specs = set()
        for _ci, _bt in enumerate(self._by_type):
            for _t in _bt:
                _specs.add((self._clip_prefix[_ci], _t))
        self._all_specs = sorted(_specs)
        # per-dataset(prefix) type registry -> pool for IN-DOMAIN (type-2) hard negatives.
        self._types_by_prefix = {}
        for _p, _t in self._all_specs:
            self._types_by_prefix.setdefault(_p, []).append(_t)

        # Flat sample index: (clip_idx, win_start, type). Positives always; a seeded
        # negative_rate fraction of empty windows added as 'none'.
        # index entries are 4-tuples (clip_idx, win_start, type, q_prefix). q_prefix=None means the
        # question uses the clip's own dataset; a non-None q_prefix marks a CROSS-DATASET hard negative
        # (foreign action asked on this window -> always the NL "no frame related" answer).
        # Build positives and negatives separately (tagged by the clip's dataset prefix) so the
        # 40% negative cap can be applied PER DATASET. Three negative kinds, all -> NL "no frame":
        #   type-1  within-clip type-absent window (natural background; abundant for fs/finegym)
        #   type-2  same-dataset OTHER type absent from the clip (in-domain disambiguation)
        #   type-3  foreign dataset's type (cross-dataset, easiest signal)
        rng = random.Random(self.seed)
        pos_by_ds, neg_by_ds = {}, {}
        for ci, rec in enumerate(self.clips):
            N = int(rec["num_frames"])
            if N <= 0:
                continue
            ds = self._clip_prefix[ci]
            clip_types = set(self._by_type[ci].keys())
            ds_types = self._types_by_prefix.get(ds, [])
            win_starts = list(range(0, N, self.stride))
            for typ, ev_frames in self._by_type[ci].items():
                for ws in win_starts:
                    end = min(ws + self.W, N)
                    if any(ws <= f < end for f in ev_frames):
                        pos_by_ds.setdefault(ds, []).append((ci, ws, typ, None))            # positive
                    elif self.negative_rate > 0.0 and rng.random() < self.negative_rate:
                        neg_by_ds.setdefault(ds, []).append((ci, ws, typ, None))            # type-1
            if self.type2_rate > 0.0 and ds_types:                      # type-2 in-domain hard negatives
                base, frac = int(self.type2_rate), self.type2_rate - int(self.type2_rate)
                for ws in win_starts:                                   # >1/window lets trimmed sets (tennis/tm,
                    k = base + (1 if rng.random() < frac else 0)        # no natural empties) reach the 40% cap
                    for _ in range(k):
                        for _try in range(8):                           # a same-dataset type absent from this clip
                            ft = ds_types[rng.randrange(len(ds_types))]
                            if ft not in clip_types:
                                neg_by_ds.setdefault(ds, []).append((ci, ws, ft, None))
                                break
            if self.cross_neg_rate > 0.0 and self._all_specs:           # type-3 cross-dataset hard negatives
                for ws in win_starts:
                    if rng.random() < self.cross_neg_rate:
                        for _ in range(8):                              # a foreign type absent from this clip
                            fp, ft = self._all_specs[rng.randrange(len(self._all_specs))]
                            if ft not in clip_types:
                                neg_by_ds.setdefault(ds, []).append((ci, ws, ft, fp))
                                break

        # per-dataset 40% negative cap: drop random excess negatives so neg <= neg_cap*(pos+neg)
        self.index = []
        caprng = random.Random(self.seed + 7)
        self.neg_stats = {}
        for ds in sorted(set(pos_by_ds) | set(neg_by_ds)):
            pos, neg = pos_by_ds.get(ds, []), neg_by_ds.get(ds, [])
            cap = self.neg_cap_by_ds.get(ds, self.neg_cap)             # per-dataset override (e.g. finegym lower)
            if 0.0 < cap < 1.0 and pos:
                max_neg = int(len(pos) * cap / (1.0 - cap))
                if len(neg) > max_neg:
                    caprng.shuffle(neg); neg = neg[:max_neg]
            self.neg_stats[ds] = (len(pos), len(neg))
            self.index.extend(pos); self.index.extend(neg)

        if balance_per_dataset:
            self.index = self._balance_index(balance_per_dataset)   # int, or per-dataset {ds: quota, 'default': q}

        self.sample_weights = self._temperature_weights() if self.temp_alpha > 0 else None

    def _temperature_weights(self):
        """Per-entry sampling weight for TWO-LEVEL temperature sampling (dataset x type).
        Dataset d sampled ~ n_d^a; within d, type t ~ n_{d,t}^a. An entry's weight is that joint
        target probability divided evenly among its n_{d,t} entries. Cross-dataset negatives are
        pooled as one pseudo-type '_cross' per clip's dataset. No data is discarded; the sampler
        just draws with these weights so finegym's 74% share drops toward its n^a share."""
        import torch
        from collections import Counter, defaultdict
        a = self.temp_alpha
        def keyof(e):
            ci, ws, typ, qp = e
            ds = self.clips[ci].get("dataset") or self._clip_prefix[ci]
            return ds, ("_cross" if qp is not None else typ)
        keys = [keyof(e) for e in self.index]
        n_dt = Counter(keys)                              # (dataset, type) -> #entries
        n_d = Counter(k[0] for k in keys)                 # dataset -> #entries
        types_of = defaultdict(set)
        for ds, tk in n_dt:
            types_of[ds].add(tk)
        Z_d = sum(c ** a for c in n_d.values())
        Z_t = {ds: sum(n_dt[(ds, tk)] ** a for tk in tset) for ds, tset in types_of.items()}
        w = [((n_d[ds] ** a) / Z_d) * ((n_dt[(ds, tk)] ** a) / Z_t[ds]) / n_dt[(ds, tk)]
             for (ds, tk) in keys]
        return torch.tensor(w, dtype=torch.double)

    def temperature_epoch_order(self, ep):
        """Deterministic TEMPERATURE-weighted, WITHOUT-replacement epoch order (replaces iid
        multinomial). Dataset d gets n_d = round((M_d^alpha / Z) * L) slots per epoch, filled by
        walking a fixed per-dataset shuffle with a cyclic cursor that advances across epochs -> every
        sample is visited before any repeats (full coverage), unlike iid sampling's coupon-collector
        waste. alpha=0.75 -> finegym n_d ~= M_d (>=1 full pass) while small sets get 1.5-2x. Pure
        function of (index, seed, ep): identical on every rank (clean DDP shard) and resume-safe."""
        import random as _r
        L = len(self.index)
        if L == 0:
            return []
        if not hasattr(self, "_ds_positions"):
            self._ds_positions = {}
            for pos, e in enumerate(self.index):
                ds = self.clips[e[0]].get("dataset") or self._clip_prefix[e[0]]
                self._ds_positions.setdefault(ds, []).append(pos)
        a = self.temp_alpha if self.temp_alpha > 0 else 1.0
        Z = sum(len(p) ** a for p in self._ds_positions.values())
        order = []
        for ds, positions in self._ds_positions.items():
            M = len(positions)
            n = max(1, int(round((M ** a / Z) * L)))
            shuf = positions[:]
            _r.Random((self.seed, ds).__hash__() & 0xFFFFFFFF).shuffle(shuf)
            start = (ep * n) % M
            order.extend(shuf[(start + i) % M] for i in range(n))
        _r.Random(1000 + ep).shuffle(order)
        return order

    def _balance_index(self, quota):
        """Down-sample self.index to <= `quota` (window,type) samples PER DATASET, spread as
        evenly as possible across that dataset's clips (round-robin: take window-slot 0 from every
        clip, then slot 1, ...). Maximizes clip diversity within the quota; a dataset with fewer
        clips than the quota simply uses more windows per clip (all it has).

        `quota` may be an int (same cap for every dataset) or a dict {dataset: quota, 'default': q}
        for per-dataset caps (e.g. finegym 640 / others 200 at eval). A dataset absent from the dict
        with no 'default' key is left uncapped."""
        rng = random.Random(self.seed + 12345)
        by_ds = {}                          # dataset -> clip_idx -> [entries]
        for e in self.index:
            ds = self.clips[e[0]].get("dataset")
            by_ds.setdefault(ds, {}).setdefault(e[0], []).append(e)
        out = []
        for ds, clips in by_ds.items():
            q = (quota.get(ds, quota.get("default", 10 ** 12)) if isinstance(quota, dict) else int(quota))
            lists = list(clips.values())
            for lst in lists:
                rng.shuffle(lst)
            rng.shuffle(lists)
            picked, slot = [], 0
            while len(picked) < q:
                progressed = False
                for lst in lists:
                    if slot < len(lst):
                        picked.append(lst[slot]); progressed = True
                        if len(picked) >= q:
                            break
                if not progressed:
                    break
                slot += 1
            out.extend(picked)
        rng.shuffle(out)
        return out

    # ------------------------------------------------------------------ #
    def _subset_per_dataset(self, records, subset, max_clips):
        """Stratified, seeded down-selection of clips, grouped by the 'dataset' key
        (a converted <split>.json is single-dataset, but grouping keeps this correct
        if several are concatenated)."""
        if subset is None and max_clips is None:
            return list(records)
        groups = {}
        for r in records:
            groups.setdefault(r.get("dataset"), []).append(r)
        kept = []
        for k in sorted(groups, key=lambda x: (x is None, x)):
            g = list(groups[k])
            random.Random((self.seed, k).__hash__()).shuffle(g)
            n = len(g)
            if subset is not None:
                n = max(1, int(round(len(g) * float(subset))))
            if max_clips is not None:
                n = min(n, int(max_clips))
            kept.extend(g[:n])
        return kept

    def _question(self, rec, typ, ci, prefix=None):
        prefix = prefix or rec["video_id"].split("/", 1)[0]   # foreign prefix for cross-dataset negs, else clip's own
        if question_for is not None:
            r = random.Random((self.seed, rec["video_id"], typ, prefix,
                               self.question_phase).__hash__() & 0xFFFFFFFF)
            try:
                return question_for(prefix, typ, phase=self.question_phase, rng=r)
            except Exception:
                pass
        pool = self._q_pool[ci].get(typ) or [""]
        return random.Random((self.seed, rec["video_id"], typ).__hash__()).choice(pool)

    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ci, ws, typ, q_prefix = self.index[i]     # q_prefix != None -> cross-dataset hard negative
        rec = self.clips[ci]
        N = int(rec["num_frames"])
        if self.jitter and N > self.W:            # temporal-crop aug: shift the window start (long clips only)
            ws = min(max(0, ws + random.randint(-self.jitter, self.jitter)), N - 1)
        end = min(ws + self.W, N)
        frames = self.fs.load_window(rec["video_id"], ws, end)     # (T,H,W,3) uint8
        T = int(frames.shape[0])
        # window-local indices of this type's events (clamp to frames loaded). For a cross-dataset
        # negative the foreign `typ` is absent from this clip -> .get returns [] -> empty target -> NL neg.
        target = [f - ws for f in self._by_type[ci].get(typ, []) if ws <= f < ws + T]
        return {
            "video_id": rec["video_id"],
            "dataset": rec.get("dataset"),
            "question": self._question(rec, typ, ci, prefix=q_prefix),
            "type": typ,
            "target_frames": target,             # [] -> 'none'
            "win_start": int(ws),
            "num_win_frames": T,
            "fps": rec.get("fps"),               # clip's own fps (int/float), not hardcoded
            "frames": frames,                    # uint8 (T,H,W,3) RGB
            "num_frames": N,
        }


def collate_windows(batch):
    """Collate for variable-length windows.

    Windows differ in T (the final window of a clip is short) and clips differ in
    H,W across datasets, so frames CANNOT be stacked into one tensor without
    padding/resizing — which is the downstream feature-extractor's job, not the
    loader's. We therefore keep every field as a per-sample list (no padding, no
    copy). ``frames`` becomes a list of (T,H,W,3) uint8 arrays. Use a plain
    DataLoader(batch_size=..., collate_fn=collate_windows); the feature module
    iterates the frames list.
    """
    if not batch:
        return {}
    keys = batch[0].keys()
    return {k: [s[k] for s in batch] for k in keys}


# --------------------------------------------------------------------------- #
# smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import subprocess
    import tempfile

    from frame_source import DirFrameSource

    DATA_ROOT = "/home/chang/Dataset/Action_Spotting"
    tmp = tempfile.mkdtemp(prefix="dpc_anns_")

    # 1) run the converter to a temp dir (module form so dpc.questions_multi resolves)
    print(f"[smoke] converting annotations -> {tmp}")
    subprocess.run(
        [sys.executable, "-m", "dpc.make_dpc_annotations", "--root", DATA_ROOT, "--out", tmp],
        cwd=_REPO, check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )

    fs = DirFrameSource(DATA_ROOT)

    # 2) HOI4D: short (300f) clips -> one window each, positive 'touch' samples
    hoi4d = os.path.join(tmp, "touchmoment_hoi4d", "train.json")
    ds = WindowedSpottingDataset(hoi4d, fs, window_frames=320, max_clips=3, seed=0)
    print("\n=== touchmoment_hoi4d/train.json  (window_frames=320, max_clips=3) ===")
    print("num samples:", len(ds))
    s = ds[0]
    print("sample keys:", list(s.keys()))
    print("video_id   :", s["video_id"], "| dataset:", s["dataset"], "| fps:", s["fps"])
    print("type       :", s["type"])
    print("question   :", s["question"])
    print("win_start  :", s["win_start"], "| num_win_frames:", s["num_win_frames"],
          "| num_frames:", s["num_frames"])
    print("frames.shape:", s["frames"].shape, s["frames"].dtype)
    print("target_frames:", s["target_frames"])

    # collate check
    batch = collate_windows([ds[i] for i in range(min(3, len(ds)))])
    print("collated frames -> list of", len(batch["frames"]), "arrays; shapes:",
          [f.shape for f in batch["frames"]])

    # 3) finegym: long clips -> MULTIPLE windows per clip (verify windowing)
    fg = os.path.join(tmp, "finegym", "train.json")
    dsf = WindowedSpottingDataset(fg, fs, window_frames=320, max_clips=2,
                                  negative_rate=0.3, seed=1)
    print("\n=== finegym/train.json  (window_frames=320, max_clips=2, negative_rate=0.3) ===")
    print("num samples:", len(dsf))
    # windows per selected clip
    for ci, rec in enumerate(dsf.clips):
        wins = sorted({ws for (c, ws, _t) in dsf.index if c == ci})
        print(f"  clip {rec['video_id']}  num_frames={rec['num_frames']}  "
              f"types={sorted(dsf._by_type[ci])}  -> {len(wins)} distinct windows "
              f"(starts {wins[:5]}{'...' if len(wins) > 5 else ''})")
    n_none = sum(1 for (c, ws, t) in dsf.index
                 if not any(ws <= f < ws + dsf.W for f in dsf._by_type[c].get(t, [])))
    print("  'none' (event-absent) samples:", n_none, "/", len(dsf.index))
    # pull one finegym sample to confirm frame decode on a long clip
    sf = dsf[len(dsf) // 2]
    print("  mid sample: type=", sf["type"], "win_start=", sf["win_start"],
          "frames.shape=", sf["frames"].shape, "target_frames=", sf["target_frames"])
    print("\n[smoke] OK")
