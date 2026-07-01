"""Central config for the touch_tsp pipeline. Edit here (or set env vars) to
port to another machine. Everything else imports paths from this file."""
import os

PROJ = os.path.dirname(os.path.abspath(__file__))

# ---- EXTERNAL inputs (large; not shipped with the code) ----
# JPG frames, one dir per video: <FRAMES_DIR>/<video>/000000.jpg ... (override per machine)
FRAMES_DIR = os.environ.get('TOUCH_FRAMES_DIR', '/data/dong/project/Workspace/dataset/hoi4d/frames')
# label splits train.json / val.json / test.json (+ class.txt). Default: the copy shipped in-repo.
LABEL_DIR  = os.environ.get('TOUCH_LABEL_DIR',  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'HOI4D-v3'))

# ---- INTERNAL (shipped) ----
DATA_DIR   = os.path.join(PROJ, 'data')                       # CSVs + label mappings
TRAIN_CSV  = os.path.join(DATA_DIR, 'hoi4d_train_tsp.csv')
VAL_CSV    = os.path.join(DATA_DIR, 'hoi4d_val_tsp.csv')
TEST_CSV   = os.path.join(DATA_DIR, 'hoi4d_test_tsp.csv')
TR_LABEL_MAP = os.path.join(DATA_DIR, 'temporal_region_label_mapping.json')
ACTION_LABEL_MAP = os.path.join(DATA_DIR, 'action_label_mapping.json')   # dual-head: touch/untouch

# ---- OUTPUTS (produced by the pipeline; default project-local) ----
# Mirrors methods/: encoders/{tsp,vjepa} front-ends, spot_head, astrm.
OUT_DIR    = os.environ.get('TOUCH_OUT_DIR', os.path.join(PROJ, 'outputs'))
TSP_OUT    = os.path.join(OUT_DIR, 'encoders', 'tsp')        # TSP encoder artifacts
GVF_PATH   = os.environ.get('TOUCH_GVF_PATH',
                            os.path.join(TSP_OUT, 'gvf', 'mvit_v1_b-max_gvf.h5'))
TRAIN_OUT  = os.environ.get('TOUCH_TRAIN_OUT', os.path.join(TSP_OUT, 'train'))
FEATURES_OUT = os.environ.get('TOUCH_FEATURES_OUT', os.path.join(TSP_OUT, 'features'))

# ---- SPOT_HEAD (spotting heads on TSP / V-JEPA features) ----
SPOT_HEAD_OUT     = os.environ.get('TOUCH_SPOT_HEAD_OUT', os.path.join(OUT_DIR, 'spot_head'))
SPOT_HEAD_CLIP_LEN        = int(os.environ.get('TOUCH_SPOT_HEAD_CLIP_LEN', 100))     # head temporal window
SPOT_HEAD_NUM_EPOCHS      = int(os.environ.get('TOUCH_SPOT_HEAD_EPOCHS', 50))

# ---- hyperparams (paper-aligned: E2E-Spot B.3) ----
CLIP_LEN   = 12          # training clip length AND feature-extraction window
FRAME_RATE = 15
GVF_DIM    = 768         # MViT-B feature dim
FEAT_DIM   = 512         # R(2+1)D-34 backbone output (spot_head feature dim)


def add_proj_to_path():
    """Make the vendored TSP code importable: `common`, `models`, and the
    dataset modules under methods/encoders/tsp/."""
    import sys
    for p in (PROJ, os.path.join(PROJ, 'methods', 'encoders', 'tsp')):
        if p not in sys.path:
            sys.path.insert(0, p)
