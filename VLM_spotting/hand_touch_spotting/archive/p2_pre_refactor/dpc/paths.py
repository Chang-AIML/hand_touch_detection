"""Central, env-configurable paths for the Action_Spotting (DPC) deployment.

Every path is `os.environ.get(<ENV>, <default>)` so the pipeline ports to another
machine by setting env vars only (no code edits). Defaults target the Deep Purple
Cluster layout under /data/Action_Spotting.

Import from here instead of hardcoding, e.g.:
    from dpc.paths import FEAT_DIR, GRID_EVEN, ANN_DIR

The hardcoded `/home/chang_noroot/...` constants scattered across the tree
(FEAT / GRID / FRAMES / LAB / _COMMON / HF_HOME / VJEPA_*) map onto the constants
below -- see dpc/README or the audit for the file:line -> constant mapping.
"""
import os

# ---- root of all Action_Spotting data on the target node --------------------
DATA_ROOT = os.environ.get('AS_DATA_ROOT', '/data/Action_Spotting')

# sqlite clip index; clip id == "<video_prefix>/<video>" (see dpc/make_dpc_annotations.py)
INDEX = os.environ.get('AS_INDEX', os.path.join(DATA_ROOT, 'index.sqlite'))

# DPC annotation splits produced by dpc/make_dpc_annotations.py:
#   <ANN_DIR>/<registry_key>/{train,val,test}.json  (schema: video_id/type/frame/question)
# Replaces the old external LAB = hand_touch_detection/data/HOI4D-v3 and the
# in-repo data/annotations/<split>.json sources.
ANN_DIR = os.environ.get('AS_ANN_DIR', os.path.join(DATA_ROOT, 'annotations_dpc'))

# ---- V-JEPA 2.1 precomputed features (one .npy per clip: <dir>/<video>.npy) --
# mean-pooled per-frame interleave motion tokens (N, 768)  -> replaces FEAT / F3 / F3DIR
FEAT_DIR = os.environ.get('AS_FEAT_DIR', os.path.join(DATA_ROOT, 'feat_interleave'))
# unpooled even grid (N/2, 576, 768) for the language-compress front-end -> replaces GRID (grid_even)
GRID_EVEN = os.environ.get('AS_GRID_EVEN', os.path.join(DATA_ROOT, 'grid_even'))
# windowed grids -> replaces GRID (grid_win)
GRID_WIN = os.environ.get('AS_GRID_WIN', os.path.join(DATA_ROOT, 'grid_win'))

# ---- raw JPG frames: <FRAMES_DIR>/<video>/000000.jpg ... --------------------
# replaces the hardcoded .../dataset/hoi4d/frames (ViT anchors read these).
# Keeps the legacy TOUCH_FRAMES_DIR env name that models/idx_localizer.py already honours.
FRAMES_DIR = os.environ.get('TOUCH_FRAMES_DIR', os.path.join(DATA_ROOT, 'frames'))

# ---- HuggingFace cache (weights/tokenizer) ----------------------------------
# replaces the hardcoded HF_HOME = /home/chang_noroot/data2/hf_cache
HF_HOME = os.environ.get('HF_HOME', os.path.join(DATA_ROOT, 'hf_cache'))

# ---- V-JEPA 2.1 encoder repo + checkpoint (feature extraction only) ---------
VJEPA_REPO = os.environ.get('VJEPA_REPO', '/opt/vjepa2')
VJEPA_CKPT = os.environ.get('VJEPA_CKPT', '/opt/vjepa2_1_vitb_dist_vitG_384.pt')

# ---- Qwen3-VL backbone (models/wrapper.py MODEL_ID) -------------------------
QWEN_PATH = os.environ.get('QWEN_PATH', 'Qwen/Qwen3-VL-8B-Instruct')

# ---- run outputs (checkpoints / metrics.csv / logs) -------------------------
OUT_DIR = os.environ.get('AS_OUT', '/data/runs')

# ---- vendored scorer -------------------------------------------------------
# The spotting-mAP scorer that used to live in the external hand_touch_detection
# repo (referenced as _COMMON) is now vendored at dpc/eval_vendor/. Import it as:
#     from dpc.eval_vendor.score import compute_mAPs
#     from dpc.eval_vendor.eval_nms import maps_quiet
# so no external repo / _COMMON sys.path insert is needed.
EVAL_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_vendor')


def as_env_summary():
    """Return the resolved paths as a dict (handy for logging a run's config)."""
    return {
        'DATA_ROOT': DATA_ROOT, 'INDEX': INDEX, 'ANN_DIR': ANN_DIR,
        'FEAT_DIR': FEAT_DIR, 'GRID_EVEN': GRID_EVEN, 'GRID_WIN': GRID_WIN,
        'FRAMES_DIR': FRAMES_DIR, 'HF_HOME': HF_HOME,
        'VJEPA_REPO': VJEPA_REPO, 'VJEPA_CKPT': VJEPA_CKPT,
        'QWEN_PATH': QWEN_PATH, 'OUT_DIR': OUT_DIR, 'EVAL_VENDOR': EVAL_VENDOR,
    }


if __name__ == '__main__':
    import json
    print(json.dumps(as_env_summary(), indent=2))
