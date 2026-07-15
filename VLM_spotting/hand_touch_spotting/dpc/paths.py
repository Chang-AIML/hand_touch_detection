"""Env-configurable paths for the Action_Spotting deployment. Every value is
os.environ.get(<ENV>, <cluster-default>) so the pipeline ports by setting env vars only;
defaults target the Deep Purple Cluster layout under /data/Action_Spotting."""
import os

DATA_ROOT  = os.environ.get("AS_DATA_ROOT", "/data/Action_Spotting")
INDEX      = os.environ.get("AS_INDEX", os.path.join(DATA_ROOT, "index.sqlite"))         # sqlite clip→frame index
ANN_DIR    = os.environ.get("AS_ANN_DIR", os.path.join(DATA_ROOT, "annotations_dpc"))    # <key>/{train,val,test}.json
HF_HOME    = os.environ.get("HF_HOME", os.path.join(DATA_ROOT, "hf_cache"))
VJEPA_REPO = os.environ.get("VJEPA_REPO", "/opt/vjepa2")
VJEPA_CKPT = os.environ.get("VJEPA_CKPT", "/opt/vjepa2_1_vitb_dist_vitG_384.pt")
QWEN_PATH  = os.environ.get("QWEN_PATH", "Qwen/Qwen3-VL-8B-Instruct")
OUT_DIR    = os.environ.get("AS_OUT", "/data/runs")                                      # ckpts / metrics.csv
