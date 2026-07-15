# Phase 2 — Experiment Recipe & Pipeline (`p2_slide_a70`)

Frozen-backbone connector for **frame-level temporal grounding of point events** across
multiple sports datasets, testing in-domain accuracy + **zero-shot OOD transfer to
finediving**. This run is the "improved method" vs the Phase-1 baseline `bal_scratch`
(which showed OOD **emerge-then-collapse**: finediving mAP@2 peaked 7.25 @step600 then
collapsed to 0.55 as the connector over-fit in-domain).

Launch: `kubectl apply -f DPC_instruction/dpc-train-scratch.yaml` (run_name `p2_slide_a70`).

---

## 1. Model (what trains)

| Component | Weight | State | Params |
|---|---|---|---|
| V-JEPA 2.1 ViT-B (`vjepa2_1_vitb_dist_vitG_384`) | frozen | motion features, tubelet=2 | frozen |
| **FrameCompress connector** | **from scratch** | **the only trainable module** | **27.5M** |
| Qwen3-VL-8B-Instruct (+ Qwen-ViT anchor branch) | frozen | language + index decoding | frozen |

Connector: V-JEPA grid feats → linear-interp to per-frame N → **8 learnable-query cross-attn
pool** → `Linear(6144→4096)` (align to Qwen embed dim). One motion token per frame.

## 2. Datasets

**In-domain (train + val):** touchmoment (HOI4D+TACO), tennis, finegym, **fs_perf (== fs_comp
data, 178 train videos)**. **OOD (eval only):** finediving. **Dropped in Phase 2:** soccernet
(2fps, second-level granularity — mismatch with the frame-precise story).

All are E2E-Spot benchmarks (arXiv 2207.10213) — our key baseline. Data at
`/data/Action_Spotting/annotations_dpc` (DPC) / `Project/vlm_deps/annotations_dpc` (local).

## 3. Hyperparameters (the receipt)

| Item | Value |
|---|---|
| Hardware | 4× A100-40GB single node, torchrun standalone, FSDP shard frozen Qwen |
| eff_batch | **64** = bs 1 × world 4 × grad_accum 16 |
| lr / warmup / steps | **3e-4 / 150 / 1500** (connector-only, no LoRA) |
| window | **600 frames**, stride 600 (non-overlapping natural tiling) |
| internal fps / anchors | fps=15; Qwen-ViT anchor every 5 s, max_side 252 |
| **temp_alpha** | **0.70** (dataset-level, shuffle-epoch without-replacement) |
| **jitter** | **30** frames (window-start; long clips only → breaks position prior) |
| negatives | negative_rate 0.15 (type-1) · type2_rate 1.5 (type-2) · cross_neg_rate 0.15 (type-3) |
| **neg_cap** | **0.40 global, 0.05 finegym** (drop random excess negatives) |
| eval | every **150** steps, `eval_windows 0` (loss-only train), `eval_max_tokens 64` |
| checkpoints | `ckpt_every 50` / `ckpt_secs 150` resume + **named `_conn_s{step}.pt` every 150 (all kept)** |

## 4. Pipeline (one training step)

```
tar frames (TarFrameSource) → 600-frame window → uint8 (T,H,W,3)
  ├─ V-JEPA ViT-B (frozen, tubelet=2) → grid → interp per-frame → 8-query cross-attn → Linear(6144→4096)
  │        = per-frame MOTION token  ─────────────┐
  ├─ Qwen-ViT anchor (frozen, 1 per 5 s)  ────────┤ interleaved
  └─ text query = GENERAL NL description (train: paraphrase-sampled; eval: fixed general) ┐
                                                  ▼
        Qwen3-VL-8B (frozen) → autoregressively generate target frame-INDEX tokens
                                                  ▼
        Loss = CE(index-token seq, GT frames);  empty target → NL "there is no frame related to the action"
        grad_accum ×16 → all-reduce connector grads → optimizer step (connector only)
```

## 5. Query = general NL description (the core Phase-2 change)

Opaque labels (`FX_back_salto_end`) are replaced by **general, transition-framed,
motion-grounded** class-level descriptions with **paraphrase diversity**, so the frozen LLM
learns the *language↔motion* mapping instead of memorizing label→phrase.
`dpc/questions_multi.py::question_for` → `dpc/action_descriptions_class.json` (48 labels).

- **train**: samples uniformly from {general} + 8 paraphrases (language diversity).
- **eval**: fixed `general` form (deterministic, E2E-Spot-comparable per-class query).
- **touchmoment**: keeps the curated HOI4D touch/untouch bank (its class descriptions were
  gymnastics-contaminated during generation → never used).
- Shared primitives for OOD transfer: salto/som→somersault, twist, takeoff/landing→start/end/Entry.

(Instance-level descriptions `action_descriptions_generated.json`, 812 comment-specific, are
NOT used as training queries — a find-all-per-type query can span multiple instances. Reserved
for a future specific/ordinal-query ablation. 15 vault entries fall back to class-level.)

## 6. Negative samples (three kinds → NL "no frame")

| Kind | Meaning | Rate | Note |
|---|---|---|---|
| type-1 | within-clip type-absent window (natural background) | 0.15 | abundant for fs/finegym (long, sparse) |
| type-2 | **same-dataset OTHER type absent from clip** (in-domain disambiguation) | 1.5/win | **new**; fills trimmed sets toward cap |
| type-3 | foreign dataset's type (cross-dataset) | 0.15 | easiest signal, don't over-weight |

**Per-dataset 40% cap** (drop random excess) prevents over-predicting "none"; **finegym capped
at 5%** — its natural per-query neg-rate is already 88.5% (32 similar types; querying one gives
mostly-absent), so heavy injection is unnecessary here, and a small finegym pool lifts positive
coverage to ~96%. Heavy "none" discrimination is deferred to the Phase-3 sliding-window eval (a
deployment where ~89% of per-type queries ARE negative). Resulting neg%: finegym 5% · fs 38.8% ·
tennis 27.8% · touchmoment 7.9% (only 2 types → no type-2).

## 7. Sampling — temperature shuffle-epoch (full coverage)

`WindowedSpottingDataset.temperature_epoch_order(ep)`: deterministic, **WITHOUT-replacement**,
dataset d gets `round(n_d^0.70 / Z · L)` slots/epoch via a cyclic cursor over a fixed per-dataset
shuffle → every sample visited before repeats (no coupon-collector waste of iid multinomial).
Identical on all ranks → clean DDP shard; resume-safe (pure function of index, seed, ep).

**Resulting exposure over 1500 steps (96k draws), total pool 53,923:**

| dataset | pool | pos | neg% | pos-coverage | passes |
|---|---|---|---|---|---|
| finegym | 36,863 | 35,020 | 5% | **96%** | 1.5× |
| fs_perf | 2,207 | 1,351 | 39% | 100% | 3.4× |
| tennis | 7,874 | 5,682 | 28% | 100% | 2.3× |
| touchmoment | 6,979 | 6,429 | 8% | 100% | 2.4× |

## 8. Eval & checkpoint selection

Training is **loss-only** (`eval_windows 0`; FSDP autoregressive eval is slow/fragile on
no-NVSwitch nodes). Every 150 steps a **named connector ckpt `p2_slide_a70_conn_s{step}.pt`**
is saved (all kept). Offline: run `--eval_only --init_fc <ckpt> --fsdp 0` (4 GPUs, each a full
independent model) → per-dataset + finediving-OOD mAP @ tol {0,1,2,4}. **Select the OOD-optimal
checkpoint** (Phase 1 lost its step-600 OOD peak to a single overwritten file — this run keeps
every ckpt to catch it).

## 9. Phase-1 → Phase-2 fixes applied

1. IN_DOMAIN drops soccernet (keeps fs_perf=fs_comp).
2. Prompt `hand-motion tokens` → `motion tokens` (HOI4D leftover; wrong prior for full-body/OOD).
3. Opaque labels → general NL descriptions + paraphrases (§5).
4. Named periodic connector ckpts + offline OOD-select (§8).
5. type-2 in-domain negatives + per-dataset 40% cap, finegym 5% (§6).
6. temperature shuffle-epoch, full coverage, alpha 0.70 (§7).
7. jitter 30 (breaks position prior on long clips).
8. eval_max_tokens 24 → 64; eval_every 100 → 150.

## 10. Known limitations (honest)

- **finegym positive coverage ~96%**, not 100% — the dominant pool exceeds its draw share at
  alpha 0.70; forcing 100% needs alpha→1.0 (abandons domain balance, risks faster OOD collapse).
- **Position prior only broken for long clips** (jitter needs N>window). Trimmed sets
  (tennis/touchmoment/finediving, ~300f) are single-window; their event-position prior is
  symmetric train/test (no benchmark harm) but survives for true open-world deployment → Phase 3
  pad/crop-jitter.
- **touchmoment negatives 7.9%** (only 2 types → in-domain type-2 impossible).
- **train non-overlap vs deploy sliding-window** gap remains for untrimmed deployment (Phase 3
  eval protocol: overlap sliding + NMS + window-prefill KV reuse).
- Distance loss (SCST/RAML) for tight-tolerance @0-2 → Phase 3.

## 11. Deferred (not in this run)

Ordinal/count question modes · instance-level specific queries · type-1 pure-background beyond
fs/finegym natural · distance loss · sliding-window benchmark eval protocol.
