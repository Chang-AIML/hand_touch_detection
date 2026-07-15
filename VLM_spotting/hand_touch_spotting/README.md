# hand_touch_spotting — NL-queried frame-level event spotting

Train a small **FrameCompress connector (~27.55M params, the only trainable module)** that maps frozen
**V-JEPA 2.1 ViT-B** motion features into a frozen **Qwen3-VL-8B** LLM, so that given a video window and a
natural-language query ("when does the hand release the object?") the model **outputs the frame index/indices**
of the event. Trained multi-dataset; evaluated in-domain and zero-shot out-of-domain (finediving).

## How it works (one forward pass)
```
frames (T,H,W,3) ──V-JEPA even-pass──▶ grid (⌈T/2⌉,576,768)
    │                                        │
    │  per-second frozen Qwen-ViT anchors    │ FrameCompress: interp→N, LN,
    │                                        ▼  K=8 queries ⟵ 576 patches, →d_llm, RMS-match
    └────────────▶  token sequence:  [instr][anchor][idx"f"][motion_f]…[question][answer]
                                              │
                                     frozen Qwen3-VL ──greedy generate──▶ "12, 47" ──parse──▶ frames
```
Only `FrameCompress` is trained (teacher-forced CE on the answer digits). Everything else is frozen.

## Layout
```
train.py                 entry shim -> dpc.train_mixed.main
dpc/
  train_mixed.py         driver: argparse, DDP/FSDP, train loop, eval loop, checkpoint/resume, eval_only, diag
  windowed_dataset.py    clips -> fixed windows -> (query,target) samples; 3 negative kinds; temperature-epoch sampler
  questions.py           label -> natural-language query (HOI4D bank / class descriptions / templated fallback)
  frame_source.py        TarFrameSource (cluster, tar+sqlite) / DirFrameSource (local jpg)
  vjepa.py               frozen V-JEPA 2.1 even-pass grid extractor (on the fly)
  paths.py               env-configurable paths (AS_DATA_ROOT, AS_ANN_DIR, AS_OUT, QWEN_PATH, VJEPA_*, HF_HOME)
  eval_vendor/           point-event mAP@tol scorer: score.py (numeric core) + eval_nms.maps_quiet
models/
  frame_compress.py      the trainable connector (state_dict keys are a checkpoint contract — see below)
  localizer.py           builds the token sequence, greedy-generates + parses frames, teacher-forced loss
  wrapper.py             frozen Qwen3-VL boundary (tokenizer, embeds, generate, ViT anchors)
tests/                   verification harness (see below)
archive/                 superseded material (old experiments/docs) + p2_pre_refactor/ (pre-refactor source)
```

## Run
Set the env (`VJEPA_REPO`, `VJEPA_CKPT`, `QWEN_PATH`, `AS_DATA_ROOT`, `AS_ANN_DIR`, `AS_OUT`, `HF_HOME`).

- **Train (cluster, 4×A100, FSDP):** `torchrun --standalone --nproc_per_node=4 dpc/train_mixed.py --run_name <name> --fsdp 1 --max_steps 1500 …` (see `/home/chang/DPC_instruction/dpc-train-scratch.yaml`).
- **Eval a connector (local):** `bash outputs/local_eval/run_rigorous.sh <conn_sXXX.pt>` (or `run_ood.sh` / `run_indom.sh`).
- **Locally:** `python train.py …` works too (same driver).

## Extend
- **New connector variant:** implement `forward(grid, text_emb, N) -> (N, d_llm)` + `set_target_rms_from(embed_weight)`; pass it as `Localizer(W, <your_connector>, …)` in `train_mixed.py`.
- **New dataset:** add its `<key>/{train,val,test}.json` under `AS_ANN_DIR` and its label→description in `questions.py` / `action_descriptions_class.json`. No model-core change.
- **New LLM backbone:** re-implement the `QwenWrapper` surface (`tokenizer, embed_tokens, model, llm, core, d_llm, n_layers, vit_anchor_groups`).

## Invariants (do not break)
- **Checkpoint contract:** `FrameCompress` state_dict keys `q, in_ln.*, a3.n.*, a3.a.*, out.0.*, out.1.*` — existing `conn_s*.pt` load `strict=True`. Never rename these attributes.
- **DPC parity:** the cluster YAMLs pin the entry path `dpc/train_mixed.py` and repo at `/data/code/hand_touch_spotting`. Deprecated flags (`--stain/--use_text/--gate_lang/--lora*`) are kept parseable (asserted 0) so stale YAMLs still run.
- **Numeric:** the prompt literals in `localizer.py` (marked `# INVARIANT`), the token layout / anchor cadence, greedy decoding, the fp32 V-JEPA even-pass, and `eval_vendor/score.py` all determine the mAP — changing any shifts the numbers.
- **Note:** touchmoment eval questions are seeded from a string-containing tuple hash, so their exact phrasing (and thus touchmoment mAP) depends on `PYTHONHASHSEED`. Pin it for reproducible touchmoment numbers.

## Verify (behavior-preserving refactor guarantee)
`tests/run_all.sh` runs: import + `--help` + verbatim-literal smoke, `conn_*.pt` strict state_dict load,
CLI back-compat (the real YAML/script flag vectors parse), dataset determinism, and the **golden eval**
(`golden_eval.sh`: a fixed connector on a tiny in-domain+OOD set, greedy, `PYTHONHASHSEED=0`) — the per-dataset
mAP@tol must match `tests/golden/golden.txt` exactly. `tests/train_smoke.sh` runs one training step end-to-end.
