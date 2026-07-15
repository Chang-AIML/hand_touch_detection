# `p2_slide_a70` — Data Recipe and Distribution

All figures below are measured from the annotation files actually used by this run, not estimated.

Annotation root (training): `vlm_deps/annotations_dpc/` (on DPC: `/data/Action_Spotting/annotations_dpc`).
Registry key `fs_perf` holds the **fs_comp** data: the 178 train video ids match 178/178 after stripping
the `fs_perf/` prefix. (`fs371` = 178 train + 59 val + 134 test.)

`finediving` is **held out** — it appears only in evaluation. Its `fps` field was corrected from `-1` to `30`.

---

## 1. Raw split statistics (train)

| dataset | clips | events | types |
|---|---|---|---|
| touchmoment | 3234 | 13325 | 2 |
| tennis | 1368 | 10976 | 6 |
| finegym | 3327 | 50122 | 32 |
| fs_perf | 178 | 1764 | 4 |
| finediving | — (held out) | — | 4 |

## 2. Clip-length distribution (relative to the 600-frame window)

| dataset | split | clips | min | median | max | clips > 600 f |
|---|---|---|---|---|---|---|
| touchmoment | train | 3234 | 73 | 300 | 412 | 0 (0 %) |
| touchmoment | test | 649 | 81 | 300 | 324 | 0 (0 %) |
| tennis | train | 1368 | 164 | 307 | 1195 | 91 (7 %) |
| tennis | test | 1622 | 178 | 363 | 1442 | 181 (11 %) |
| finegym | train | 3327 | 193 | 1062 | 5225 | 2060 (62 %) |
| finegym | test | 1103 | 208 | 1091 | 4566 | 701 (64 %) |
| fs_perf | train | 178 | 3975 | 4275 | 4375 | 178 (100 %) |
| fs_perf | test | 134 | 4150 | 4275 | 4350 | 134 (100 %) |

`touchmoment` and most of `tennis` are single-window clips. `finegym` clips are long (median 1062 f) and
`fs_perf` clips are full performances (~4275 f ≈ 2.4 min @ 30 fps).

## 3. Natural window structure (600-frame non-overlapping tiling)

"Empty" = a window containing **no event of any type**.

| dataset | split | videos | windows | action windows | empty windows | empty % |
|---|---|---|---|---|---|---|
| touchmoment | train | 3234 | 3234 | 3229 | 5 | 0.2 % |
| touchmoment | test | 649 | 649 | 649 | 0 | 0.0 % |
| tennis | train | 1368 | 1459 | 1412 | 47 | 3.2 % |
| tennis | test | 1622 | 1813 | 1726 | 87 | 4.8 % |
| finegym | train | 3327 | 9525 | 8598 | 927 | 9.7 % |
| finegym | test | 1103 | 3233 | 2863 | 370 | 11.4 % |
| fs_comp | train | 178 | 1408 | 608 | 800 | **56.8 %** |
| fs_comp | test | 134 | 1057 | 454 | 603 | **57.0 %** |
| finediving | train | 1801 | 1801 | 1801 | 0 | 0.0 % |
| finediving | test | 749 | 749 | 749 | 0 | 0.0 % |

Train and test empty fractions agree within each dataset.

**Per-query view (finegym).** The table above counts "any event". Because a query asks for **one** type,
the natural negative rate per (window, type) is much higher: over all 32 types,
pos = 35,020 and neg = 269,780 → **88.5 % of (window, type) pairs are naturally negative**.

## 4. Per-class distribution (train)

| dataset | total events | types | events per class (avg) |
|---|---|---|---|
| touchmoment | 13325 | 2 | 6662 |
| tennis | 10976 | 6 | 1829 |
| finegym | 50122 | 32 | 1566 |
| fs_perf | 1764 | 4 | 441 |

**finegym is internally very imbalanced**: min 23, median 1263, max 4143 events per class.

Rarest finegym classes: `FX_side_salto_start` 23, `FX_side_salto_end` 23, `FX_turns_start` 682,
`FX_turns_end` 682, `UB_dismounts_start` 750, `UB_dismounts_end` 750.

---

## 5. Query construction

- Mode: **find-all-per-type** only ("find every frame of type X in this window").
  Ordinal and count modes exist in the code but are not wired into windowed training.
- Query text = **class-level natural-language description** of the action, generated once per label
  (48 labels, 0 failures) with 8 paraphrases each.
  - **train**: uniformly sample from {general} ∪ {8 paraphrases}
  - **eval**: the fixed `general` sentence
  - **touchmoment**: overridden by the curated HOI4D touch/untouch question bank
- 812 distinct **instance-level** (dataset, label, cleaned-comment) descriptions were also generated
  (797 valid, 15 failures on long vault comments). **They are not used as training queries** — a
  find-all-per-type query may span several instances with different comments.
- Negative answer text: `"there is no frame related to the action"` (natural language, not `none`).

## 6. Negative sample design

Three kinds, all with an empty target:

| kind | definition | rate |
|---|---|---|
| type-1 | a type present in the clip but absent from **this** window | `negative_rate` = 0.15 |
| type-2 | another type of the **same dataset**, absent from the clip | `type2_rate` = 1.5 per window |
| type-3 | a type from a **different dataset**, absent from the clip | `cross_neg_rate` = 0.15 |

Type-1 can only occur for clips longer than one window (finegym, fs_perf).

**Per-dataset cap** applied afterwards (random excess dropped): 0.40 globally, **0.05 for finegym**.

## 7. Final training pool (the index actually used)

| dataset | positives | negatives | neg % | pool |
|---|---|---|---|---|
| touchmoment | 6429 | 550 | 7.9 % | 6979 |
| tennis | 5682 | 2192 | 27.8 % | 7874 |
| finegym | 35020 | 1843 | 5.0 % | 36863 |
| fs_perf | 1351 | 856 | 38.8 % | 2207 |
| **total** | **48482** | **5441** | **10.1 %** | **53923** |

`touchmoment` cannot reach the 40 % cap: it has only 2 types, both usually present in a clip, so type-2
negatives are almost never constructible; its negatives come from type-3 only.

## 8. Sampling scheme and exposure

**`temperature_epoch_order`** — deterministic, **without replacement**:
dataset *d* receives `round(n_d^α / Z · L)` slots per epoch (α = 0.70, L = pool size), drawn by walking a
fixed per-dataset shuffle with a cursor that advances across epochs. Identical on every rank; resume-safe.
This replaces i.i.d. multinomial sampling, which wastes draws (coupon-collector) on repeats.

Budget: 1500 steps × eff_batch 64 = **96,000 draws** over a pool of 53,923.

| dataset | pool | positive coverage | passes (draws / pool) |
|---|---|---|---|
| touchmoment | 6979 | 100 % | 2.39× |
| tennis | 7874 | 100 % | 2.30× |
| finegym | 36863 | **95.9 %** | 1.46× |
| fs_perf | 2207 | 100 % | 3.39× |

**Per-type coverage check (finegym, 32 types).** No type receives zero samples. The two rarest classes
(`FX_side_salto_start/end`, 23 positives each) are covered 21/23 = **91.3 %**; 91.3 % is the minimum
per-type coverage across all 32 types. Coverage is spread randomly because the per-dataset shuffle is
random, so the uncovered 4.1 % of finegym is not concentrated in any class.

**Windows.** 600 frames, stride 600 (non-overlapping). `jitter` = 30 frames on the window start, which
only takes effect for clips longer than the window (finegym, fs_perf, part of tennis).

---

## 9. Evaluation set composition

Balances used: 200 (tennis / fs_perf / touchmoment), 640 (finegym), **1018 = full val (finediving)**.
Subsampling is a seeded, stratified round-robin over clips (deterministic, not i.i.d. random).

| dataset | positives | empty (negative) windows | empty % | total | val pool | sampled fraction |
|---|---|---|---|---|---|---|
| touchmoment | 200 | 0 | 0.0 % | 200 | 276 | 72.5 % |
| tennis | 199 | 1 | 0.5 % | 200 | 1894 | 10.6 % |
| finegym | 585 | 55 | 8.6 % | 640 | 9715 | 6.6 % |
| fs_perf | 137 | 63 | **31.5 %** | 200 | 452 | 44.2 % |
| **finediving (OOD)** | 1018 | **0** | **0.0 %** | 1018 | 1018 | **100 %** |

in-val total = 1240 samples; ood-val = 1018 samples.

Underlying validation pools (600-f tiling, (window, type) positives):

| dataset | val clips | (window,type) positives | gt events | types |
|---|---|---|---|---|
| touchmoment | 138 | 276 | 647 | 2 |
| tennis | 455 | 1894 | 3642 | 6 |
| finegym | 944 | 9715 | 13866 | 32 |
| fs_perf | 59 | 452 | 584 | 4 |
| finediving | 450 | 1018 | 1063 | 4 |

**Empty windows exist only where clips exceed one window.** `finediving` clips are 111 frames — one
window each, always containing the queried event — so the OOD evaluation has **no background windows**.

---

## 10. Known data issues

1. `touchmoment` class-level descriptions were generated with a gymnastics context
   ("gymnast … apparatus … beam, bars, or floor"). They are **never used**: touch/untouch fall back to
   the curated HOI4D bank.
2. `fs_perf | jump_landing` class description says "a gymnast's feet … the ground" instead of
   "a skater … the ice". It **is** used.
3. 15 of the 812 instance-level descriptions failed generation (long `finegym` VT comments, output
   truncated past `max_tokens`). Those events fall back to class-level descriptions.
4. `finediving` `fps` was `-1` in the source annotations and was corrected to `30`.
5. The training prompt previously read "per-frame **hand-motion** tokens" (a leftover from the original
   HOI4D task); this run uses "per-frame motion tokens".
6. `soccernet_ball` and `soccernetv2` are excluded from this run (2 fps, second-level annotation).
