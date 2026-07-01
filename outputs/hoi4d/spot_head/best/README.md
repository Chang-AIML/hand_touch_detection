# V-JEPA MS-TCN — best-epoch raw test predictions (for your own NMS / Soft-NMS)

One file per adapter mode, from the **best-by-val checkpoint**, evaluated on the **test** split
(424 videos). Best epochs: interleave=34, even=39, odd=36, stack=49.

## Files

- `<mode>.recall.json.gz` — **the raw file to post-process.** Gzipped JSON, high-recall dense
  per-frame predictions: every (frame, class) whose score ≥ 0.01. This is the input to NMS/Soft-NMS
  (no NMS applied yet). Format:
  ```
  [ { "video": "<id>", "events": [ {"label": "touch"|"untouch", "frame": <int>, "score": <float>}, ... ] }, ... ]
  ```
- `<mode>.argmax.json` — sparser argmax events (one per frame where argmax≠background); not needed
  for NMS but included for reference.

## Ground truth & metric

- truth: `../../../data/HOI4D-v3/test.json` (fields: video, num_frames, events:[{frame,label}]).
- mAP@δ = mean AP over classes at temporal tolerance δ; the paper-style "mAP" = mean over δ∈{0,1,2}.

## Compute NMS / Soft-NMS yourself

Everything is vendored in the repo:

```python
import gzip, json, sys
sys.path.insert(0, 'downstream/lib')                     # from repo root
from util.eval import non_maximum_supression             # hard NMS
from util.score import compute_mAPs
from downstream.eval_nms import soft_nms                  # Gaussian temporal soft-NMS

truth = json.load(open('data/HOI4D-v3/test.json'))
pred  = json.load(gzip.open('outputs/downstream/best/interleave.recall.json.gz'))

no_nms = compute_mAPs(truth, pred, tolerances=[0,1,2])                       # without NMS
nms    = compute_mAPs(truth, non_maximum_supression(pred, window=1), tolerances=[0,1,2])
snms   = compute_mAPs(truth, soft_nms(pred, window=4, sigma=0.5), tolerances=[0,1,2])
```

Or just re-run with any params:
```
python downstream/eval_nms.py --split test --tolerances 0 1 2 [--per-class] \
    --nms-window <w> --snms-window <w> --snms-sigma <s>
```
(reads these same best-epoch predictions from `outputs/downstream/vjepa_mstcn_<mode>/`.)
