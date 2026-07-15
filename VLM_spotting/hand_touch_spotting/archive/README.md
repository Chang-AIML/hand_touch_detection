# archive/ — superseded material (2026-07-10 refactor)

The project was pared down to only what the **`p2_slide_a70` PHASE2 report** needs
(`outputs/local_eval/PHASE2_FULL_REPORT.md` + `DATA_RECIPE_AND_DISTRIBUTION.md`).
Nothing here is deleted — it was **moved** (reversible) and is also inside the full backup.

## Full backup (everything, pre-refactor)
`/home/chang/Project/VLM_spotting_FULLBACKUP_2026-07-10.tar.zst` (~9 GB, zstd, integrity-verified).
Restore: `tar -I zstd -xf VLM_spotting_FULLBACKUP_2026-07-10.tar.zst -C /path`.

## What's here
- `code/` — 80 old-experiment `.py` (idx/probe/refine/action/rlvr/spot/phase0-1 scripts,
  old datasets/heads/metrics). None are imported by the p2 pipeline (verified: `train_mixed.py --help`
  imports cleanly with these removed). Package `__init__.py` markers were left in place.
- `docs/` — 8 pre-p2 planning/handoff docs (TECHNICAL_REPORT, exp_plan, *_HANDOFF, PROJECT_SUMMARY, …).

## What was KEPT in the live tree (the p2 pipeline)
- Code (21-file import closure): `dpc/{train_mixed,windowed_dataset,questions_multi,frame_source,vjepa_online,paths}.py`,
  `dpc/eval_vendor/*`, `models/{frame_compress,idx_localizer,vjepa_adaptor,wrapper}.py`, `data/{questions,token_layout,vjepa_grid,vjepa_interleave}.py`.
- Weights: `outputs/local_eval/conn_s{150..1500}.pt` (p2 trajectory) + `conn_MIX*.pt` (mechanistic 2×2),
  and — kept by request — `bal_a600_ckpt.pt` / `scratch_ckpt.pt` (pre-p2) and `outputs/plot/`.
- Reports + eval scripts + results under `outputs/local_eval/`.

## What was DELETED (in backup only; regenerable compute artifacts)
`outputs/{probe,feat_cache,idx_compress,spot,phase1,idx,idx_rlvr,action,vjepa_feats,feat_logs,logs}`
(~6.5 GB), the old handoff tarball, offline-wandb dir, `__pycache__`.
