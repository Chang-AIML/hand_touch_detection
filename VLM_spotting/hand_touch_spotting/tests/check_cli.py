"""Tier-2 CLI back-compat: the exact flag vectors from the live DPC YAML + local eval scripts
must still parse (dead flags kept as no-op). Requires dpc.train_mixed.build_parser() (added in refactor)."""
import sys
sys.path.insert(0, ".")
from dpc.train_mixed import build_parser

VECTORS = {
    # dpc-train-scratch.yaml (the LIVE 4-GPU training job)
    "train-scratch": "--run_name p2_slide_a70 --wandb_mode online --balance 0 --eval_balance 60 "
        "--window_frames 600 --jitter 30 --negative_rate 0.15 --cross_neg_rate 0.15 --type2_rate 1.5 "
        "--neg_cap 0.40 --neg_cap_finegym 0.05 --temp_alpha 0.70 --batch_size 1 --grad_accum 16 --lr 3e-4 "
        "--warmup 150 --n_q 8 --stain 0 --use_text 0 --gate_lang 0 --use_anchor 1 --anchor_stride 5 "
        "--anchor_max_side 252 --vjepa_bf16 1 --lora 0 --fsdp 1 --num_workers 8 --max_steps 1500 "
        "--eval_every 150 --eval_windows 0 --eval_max_tokens 64",
    # run_ood.sh / run_indom.sh / run_rigorous.sh (local eval)
    "eval-rigorous": "--run_name p2_rig_eval --wandb_mode offline --init_fc x.pt --fsdp 0 --eval_only 1 "
        "--local_frames /d --window_frames 600 --batch_size 1 --n_q 8 --stain 0 --use_text 0 --gate_lang 0 "
        "--use_anchor 1 --anchor_stride 5 --anchor_max_side 252 --vjepa_bf16 1 --eval_max_tokens 64 "
        "--tols 0,1,2,4,8,16 --eval_balance 200 --eval_balance_finegym 640 --eval_balance_ood 1018 --eval_windows 4000",
    # run_diag.sh
    "eval-diag": "--run_name p2_diag --wandb_mode offline --init_fc x.pt --fsdp 0 --eval_only 1 "
        "--diag_rollout 40 --diag_G 8 --diag_temps 0.7,1.0,1.3 --eval_balance 60 --eval_balance_ood 60",
}
p = build_parser()
for name, v in VECTORS.items():
    p.parse_args(v.split())  # raises SystemExit on unknown/removed flag
    print(f"  OK parse: {name}")
print("CLI PASS: all live flag vectors parse")
