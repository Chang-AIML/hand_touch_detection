# Mixed Multi-Dataset Spotting — Training Handoff (portable, non-cluster)

Reproduce the **balanced multi-dataset event-spotting** training on any machine with GPUs.
Written for a **2×RTX Pro 6000 workstation** (run directly with `torchrun`, no Slurm/k8s,
dataset already on local disk). All code lives under `hand_touch_spotting/dpc/`.

---

## 1. What this is (architecture)

Open-vocabulary **point-event spotting**: given a natural-language question about an event
("When does a serve happen?") and a video window, the model **generates the frame index** of
the event(s) as text. Trained across 6 datasets at once (domain-balanced), tested in-domain
and out-of-domain.

**Model stack (only FrameCompress is trained in Stage 1):**
```
window frames (uint8 T,H,W,3)
  → V-JEPA 2.1 ViT-B         [FROZEN, fp32]  on-the-fly, no precompute
      → grid (ceil(T/2),576,768)
  → FrameCompress            [TRAINABLE, 27.5M, question-conditioned]
      → N "motion tokens" (4096-d, one per frame)
  → Qwen3-VL-8B-Instruct     [FROZEN, bf16; idx-decode]
      → generates the frame-index digit string  ("57, 178"  or "none")
```
- Each motion token is preceded by its frame-index as digit tokens (the "copy handle",
  `use_idx=True`). Sequence length ≈ N + ~1.7·N + text.  (600 frames ≈ 2350 tokens; 512 ≈ ~2000.)
- **ViT anchors are OFF** (`use_anchor=False`) — we never read loose jpgs for anchors.
- Metric = spotting **mAP@δ** (δ∈{0,1,2,4} frames), computed per dataset **and** overall,
  reported for in-domain val and out-of-domain (finediving) val. Test split held out for final.

**Two-stage (LLaVA-style):**
| Stage | trained | Qwen | start from |
|---|---|---|---|
| **1** (current) | FrameCompress only | frozen | `comp_notext` HOI4D connector |
| **2** (next) | FrameCompress **+ LoRA** | unfrozen (LoRA) | Stage-1 best ckpt (`--lora 1 --init_fc <s1_best>`) |

---

## 2. Repo layout (`hand_touch_spotting/dpc/`)

| file | role |
|---|---|
| `questions_multi.py` | label → natural-language question (train/test phrasing split); HOI4D curated, others templated |
| `make_dpc_annotations.py` | convert raw split json → training schema + questions; encodes the in/out-domain **REGISTRY** |
| `frame_source.py` | `DirFrameSource` (loose jpgs — **use this on the workstation**) / `TarFrameSource` (tar+sqlite, cluster) |
| `windowed_dataset.py` | slice clips into ≤W-frame windows; **`balance_per_dataset`** = domain-balanced sampling |
| `vjepa_online.py` | frozen `OnlineVJEPA.extract_grid()` — bit-exact reproduction of the offline grid, computed in the loop |
| `eval_vendor/` | vendored spotting scorer (`maps_quiet`, `compute_mAPs`) — **self-contained, no external repo** |
| `paths.py` | env-configurable paths (DATA_ROOT, ANN_DIR, VJEPA_REPO/CKPT, QWEN_PATH, OUT_DIR) |
| `train_mixed.py` | the trainer: DDP + grad-accum + on-the-fly grid + parallel per-dataset eval + wandb |

Also needed from the repo: `models/` (wrapper, frame_compress, idx_localizer, vjepa_adaptor),
`data/vjepa_grid.py` + `data/vjepa_interleave.py` (V-JEPA extraction the online module reuses).

---

## 3. Dependencies (pin these — validated)

```
python 3.11+
torch (CUDA build for your GPUs; validated on 2.5.1-cu121 and 2.9.1-cu128)
transformers==4.57.1        # has Qwen3-VL
timm==1.0.27
wandb accelerate safetensors einops pillow numpy
```
Qwen3-VL needs a recent transformers; do not downgrade below 4.57.

---

## 4. Weights & assets to obtain (once)

| asset | where | put at (env) |
|---|---|---|
| **Qwen3-VL-8B-Instruct** (~17G) | HF `Qwen/Qwen3-VL-8B-Instruct` (`huggingface-cli download`) | `QWEN_PATH` (local dir) |
| **V-JEPA 2.1 ViT-B ckpt** (1.6G) | `vjepa2_1_vitb_dist_vitG_384.pt` | `VJEPA_CKPT` |
| **V-JEPA repo** (code, 14M) | `repos/vjepa2` | `VJEPA_REPO` |
| **base connector** `comp_notext/best.pt` (106M) | from this repo's `outputs/idx_compress/comp_notext/` | `--init_fc` |

The eval scorer is already vendored in `dpc/eval_vendor/` — **no external `hand_touch_detection` needed.**
On the workstation these can just live in local dirs; set the env vars in §7.

---

## 5. Data organization

**Raw split annotations** (per dataset, one json per split) must be in this unified schema:
```json
[{"video":"<clip>", "num_frames":300, "events":[{"frame":57,"label":"touch"}, ...],
  "fps":15, "width":854, "height":480}]
```
**Frames on disk** (loose jpgs, for `DirFrameSource`):
```
<DATA_ROOT>/<dataset>/<framesdir>/<clipname>/*.jpg
   framesdir = "Frames" for dataset "TouchMoment", else "frames"
   video_id  = "<dataset>/<clipname>"   (frame files sorted naturally)
```
**Step A — convert annotations** (adds NL questions + in/out-domain split; edit the `REGISTRY`
in `make_dpc_annotations.py` if your dataset keys/paths differ):
```bash
python -m dpc.make_dpc_annotations --root <DATA_ROOT> --out <ANN_DIR>
```
Produces `<ANN_DIR>/<key>/{train,val,test}.json` (nested) + `_samples.json`.
`REGISTRY` marks **finediving = out-of-domain (val/test only, never trained)**; in-domain =
TouchMoment(HOI4D+TACO), tennis, finegym, soccernet_ball, fs_perf. (fs_comp / native-TouchMoment
excluded — annotations exist but no extracted frames.)

**Balanced sampling** (in `windowed_dataset.py`): `balance_per_dataset=N` caps each dataset to
~N (window,type) samples, spread round-robin across its clips. Fixes finegym dominance (raw 41k
windows → capped) and small-set starvation (soccernet uses all 4 clips). Windowing is a fixed
non-overlapping grid (NOT centered on GT); `negative_rate` keeps a fraction of event-absent
windows as `"none"` so the model learns background.

---

## 6. Frame source: use `DirFrameSource` on the workstation

The cluster packs frames into tar shards + a sqlite offset index (`TarFrameSource`). On a
workstation with the raw dataset, just use `DirFrameSource` (loose jpgs) — set env
`AS_DATA_ROOT=<your frames root>` and pass `--local_frames <same root>` to `train_mixed.py`
(that flag switches the trainer from Tar to Dir). No index/tar needed.

---

## 7. How to run on 2×Pro 6000 (no Slurm — direct torchrun)

```bash
cd hand_touch_spotting
export VJEPA_REPO=/path/to/repos/vjepa2
export VJEPA_CKPT=/path/to/vjepa2_1_vitb_dist_vitG_384.pt
export QWEN_PATH=/path/to/Qwen3-VL-8B-Instruct
export AS_DATA_ROOT=/path/to/Action_Spotting          # your local frames root
export AS_ANN_DIR=/path/to/annotations_dpc            # output of make_dpc_annotations
export AS_OUT=/path/to/runs
export HF_HOME=/path/to/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # avoids fragmentation OOM
export WANDB_API_KEY=<your key>                        # or WANDB_MODE=offline
export PYTHONUNBUFFERED=1

# ---- Stage 1 (connector only) ----  2 GPUs -> --nproc_per_node=2
torchrun --standalone --nproc_per_node=2 dpc/train_mixed.py \
  --run_name mixed_s1 --wandb_mode online --wandb_project dpc_mixed_spotting \
  --local_frames "$AS_DATA_ROOT" \
  --init_fc /path/to/comp_notext_best.pt --stain 0 --use_text 0 --n_q 8 \
  --window_frames 600 --batch_size 1 --grad_accum 16 --lr 2e-4 --warmup 20 \
  --balance 2500 --eval_balance 60 --epochs 3 --max_steps 400 \
  --eval_every 100 --eval_windows 400 --negative_rate 0.15

# ---- Stage 2 (unfreeze LLM via LoRA), from Stage-1 best ----
torchrun --standalone --nproc_per_node=2 dpc/train_mixed.py \
  --run_name mixed_s2 --local_frames "$AS_DATA_ROOT" \
  --init_fc "$AS_OUT/mixed/mixed_s1_best.pt" --lora 1 --lora_rank 16 \
  --stain 0 --use_text 0 --n_q 8 \
  --window_frames 384 --batch_size 1 --grad_accum 16 --lr 2e-4 --lr_lora 1e-4 --warmup 20 \
  --balance 2500 --eval_balance 60 --epochs 3 --max_steps 400 --eval_every 100 --eval_windows 400
```
**Effective batch = batch_size × num_gpus × grad_accum.** On 2 GPUs use `grad_accum=16`
(1×2×16 = 32) to match the cluster's eff-batch-32.

---

## 8. Config knobs & rationale

| flag | value | why |
|---|---|---|
| `--window_frames` | 512–600 (S1), 384 (S2) | clip cap so long clips don't overflow the VLM; **Pro 6000 has more VRAM than A100-40G → you can likely run 600 in S1 and larger batch**; S2 (LoRA backprop through Qwen) needs more memory → smaller window |
| `--balance` | 2500 | domain-balanced samples/dataset (~11.6k total ≈ 1.1 epoch @ 400 steps eff-batch 32) |
| `--grad_accum` | 16 (2 gpu) | reach eff-batch 32 with batch=1 (memory-limited by the idx-handle sequence, not GPU count) |
| `--lr / --warmup` | 2e-4 / 20 | **5e-4 made loss rise (base disrupted); 2e-4 + warmup is stable** (grad-norm ~0.3–0.8) |
| `--eval_every` | 100 | eval is generation-based (slow); 100 steps ≈ tens of min |
| `--eval_balance` | 60 | balanced val windows/dataset (bigger = less noisy mAP, slower eval) |
| `--use_anchor` (hardcoded False) | — | avoid loose-jpg anchor reads; anchors not needed |

**Pro 6000 note:** if it's the 96 GB Blackwell Pro 6000, you can raise `--batch_size` to 2–4 and
drop `--grad_accum` accordingly (fewer, faster micro-steps), and keep `--window_frames 600`.
If it's the 48 GB Ada RTX 6000, treat like A100-40G (batch 1, window ≤512 for S1).

---

## 9. Gotchas we already hit (so you don't)

1. **Parallel eval is required for DDP.** rank-0-only eval + a long barrier → NCCL 10-min
   watchdog timeout → crash. `train_mixed.py` already shards eval across ranks + all-gathers
   (init_process_group has `timeout=30min` too). Keeps all GPUs busy.
2. **OOM at large window** → `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (reclaims
   fragmentation) + reduce `--window_frames`. On 40 GB, 512 fits (batch 1); 600 was borderline.
3. **Rising loss** was `lr=5e-4` disrupting the pretrained connector → use `2e-4 + warmup`.
4. **Per-step loss is very noisy** (1.2↔2.3) — that's the 6-dataset difficulty variance, not
   instability. Watch the running avg + per-dataset mAP, not single steps.
5. **On-the-fly V-JEPA runs fp32** (bit-exact to the offline grid; the RoPE quirk needs fp32).
   It is the throughput bottleneck (~0.6 samp/s global on 4×A100). bf16 would ~2× it with slight
   drift — acceptable if you need speed (edit `OnlineVJEPA(dtype=...)` and encoder dtype).
6. **soccernet_ball has only 4 train / 1 val / 2 test clips** — its per-dataset mAP is noise;
   read the well-sized datasets (finegym/tennis/hoi4d).

---

## 10. Current status (cluster run, for reference)

Stage-1 pilot (`mixed_bal_v3`, from comp_notext, balanced, win512, 4×A100): pipeline healthy,
loss decreasing (2.2→~1.4, gnorm small), step-0 in-domain mAP@2 ≈ 3.4 (low — averages 6 datasets
incl. ones the base is zero-shot on). Tight-tolerance mAP@2 barely moved in 100 steps (expected
for frozen-LLM Stage 1); coarse mAP@4 + detection count rose. **Per-dataset eval was just added**
to see which datasets mixing actually learns. Stage-2 (LoRA) is the lever expected to move mAP@2.

Watch wandb project `dpc_mixed_spotting`, keys `val/in/<dataset>_mAP@2` and `val/ood/finediving_mAP@2`.
