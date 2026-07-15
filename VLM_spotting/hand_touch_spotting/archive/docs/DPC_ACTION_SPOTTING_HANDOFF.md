# DPC × Action_Spotting — Handoff for the Training Agent

> Everything you need to train on the **Deep Purple Cluster (DPC)** using the
> **Action_Spotting** dataset that is already uploaded to a PVC. All facts below
> were verified on the live cluster on 2026-07-05/06. Read this before writing
> any Kubernetes YAML or dataloader code.

---

## 0. TL;DR (the 6 things that will trip you up)

1. **Frames live inside tar shards + one SQLite offset index**, NOT as loose image files. Read them with `as_dataset.py` (shipped next to this doc). Do **not** try to `ls`/`open` individual `.jpg` paths on the PVC — they are packed.
2. **Annotations are loose editable files** (`open()` them normally). Format is unified: `{video, num_frames, events, ...}`.
3. **Max 4 GPUs per Job.** Requesting 5–8 gets the Job silently `Suspended` (never runs). A100 nodes are 4-GPU. For >4 GPUs you must do multi-node DDP.
4. **Two datasets have annotations but ZERO extracted frames** and cannot be trained on as-is: `fs_comp` and `TouchMoment/native`. See §5.
5. **Never put a token in YAML.** Reference the pre-existing secrets by name only (§2).
6. **Clean up Jobs** when done. Never hold a GPU idle (admins may kill it).

---

## 1. Cluster identity

- Cluster: **Deep Purple Cluster (DPC)**, AIML. Docs: https://help.cluster.aiml.team (Google-login gated).
- kubectl context: **`dpc`**. Namespace: **`chang-dong`** (already the default).
- Storage class: **`beegfs`** (ReadWriteMany, parallel FS). Storage quota: **1000Gi**.
- Container registry: **`docker.aiml.team`**.

### Local prerequisite (PATH gotcha)
The `kubectl oidc-login` plugin is installed via krew but `~/.krew/bin` must be on PATH.
It is already added to `~/.bashrc`, but if `kubectl` errors with `unknown command "oidc-login"`, run:
```bash
export PATH="$HOME/.krew/bin:$PATH"
```
Re-auth (opens browser, @aiml.team Google login): `python3 ~/Downloads/dpc-authenticator/dpc_setup.py`
(the cached OIDC token refreshes silently once the plugin is on PATH).

---

## 2. Auth & secrets (do NOT recreate these; do NOT ask for tokens)

Two secrets already exist in the `chang-dong` namespace. Reference them by name only:
- **`gitlab-docker-secret`** (dockerconfigjson) → pull private images from `docker.aiml.team`. Use via `imagePullSecrets`.
- **`gitlab-token`** (key `access-token`) → clone private repos. Inject as env `GITLAB_TOKEN`.

**Security rule (hard):** never generate, echo, log, or embed an access token/password/kubeconfig in any command, YAML, or example. Keep `<username>`/`<project>` as placeholders unless given concrete non-secret values.

---

## 3. Hardware & GPU constraints (verified via `nvidia-smi`)

| Fact | Value |
|---|---|
| A100 model | **A100-SXM4-40GB** |
| A100 node topology | **4 GPUs/node, direct NVLink (NV4 ≈ 200 GB/s bi-dir per pair), NO NVSwitch** |
| **Max GPUs per Job** | **4** — requesting 5/8 makes the Job `Suspended` and it never runs (global admission cap) |
| Inter-node network | 2× Mellanox `mlx5` InfiniBand NICs (for multi-node DDP) |
| Other GPU types | V100(DGX) 32GB, Ada A6000 48GB, L40S 48GB |
| CUDA | 12.2+ (use CUDA 12.x base images) |

**Implication:** single-Job training is capped at **4×A100** (NVLink-connected, great for NCCL). To use more, run N Jobs/pods each ≤4 GPU and coordinate with PyTorch DDP over InfiniBand, or email `admins@aiml.team` for a special arrangement.

---

## 4. The dataset on the PVC (this is the important part)

- PVC: **`action-spotting-pvc`** (800Gi, beegfs). Mount it at **`/data`**.
- Root: **`/data/Action_Spotting/`** (~632G). Layout:

```
/data/Action_Spotting/
├── <dataset>/
│   ├── frames/  <dataset>-00000.tar, -00001.tar, …   ← frames, packed by clip (~1.5GB/shard)
│   └── train.json, val.json, test.json, class.txt …  ← loose annotations (editable)
├── … (finegym, soccernetv2, soccernet_ball, tennis, fs_perf, finediving, fs_comp, TouchMoment)
└── index.sqlite   (6.1G)   ← offset index over ALL frame tars
```

### index.sqlite — PURE FRAMES
Table `files(path, shard, offset, size, clip, ord)`, indexes `idx_path`, `idx_clip`.
- **18,633,450 frames / 17,120 clips** (images only; annotations are NOT in the index).
- `path`  = e.g. `finegym/<clip>/000001.jpg` (dataset-prefixed)
- `shard` = e.g. `finegym/frames/finegym-00000.tar` (relative to root)
- `offset`,`size` = byte range of that frame's JPEG inside the shard
- `clip`  = e.g. `finegym/<clipname>` (soccernet nests deeper, e.g. `soccernetv2/england_epl/2014-2015/<game>`)
- `ord`   = frame order within the clip (0-based)

### How to read frames (use the shipped `as_dataset.py`)
Random access is a byte `seek` into the shard → ~5 ms/frame, no unpacking, no loose files.
```python
from as_dataset import TarStore, ClipDataset   # ships next to this doc

# (a) arbitrary frame or file, by path — like open(folder/path)
store = TarStore("/data/Action_Spotting", "/data/Action_Spotting/index.sqlite")
jpg_bytes = store.read("finegym/<clip>/000001.jpg")

# (b) PyTorch Dataset: each item = one clip's frames (decoded)
ds = ClipDataset("/data/Action_Spotting", "/data/Action_Spotting/index.sqlite", transform=...)
sample = ds[0]                 # {'clip': 'finegym/<clip>', 'frames': [tensor/PIL, ...]}
# DataLoader(ds, num_workers=8, ...) is fork-safe (handles opened lazily per worker).
# Training image MUST have Pillow (+ torch) for ClipDataset.
```
`TarStore.clips()` lists all 17,120 clip ids (~4.5 s one-time). `TarStore.clip_frames(clip)` returns `[(path, shard, offset, size), …]` ordered by `ord`.

### How to read annotations (plain files)
```python
import json
train = json.load(open("/data/Action_Spotting/finegym/train.json"))
```

### ⚠️ Frames are immutable; annotations are editable
The frame tars + index never change → **the index never needs rebuilding.** You may freely edit/add loose annotation files. If you ever DO change frame tars, rebuild with `build_index_parallel.py` (see §8) — but you shouldn't need to.

---

## 5. Dataset statistics (verified) — and what is NOT trainable

### Total frames per dataset
| dataset | frames | trainable? |
|---|--:|:--:|
| finegym | 7,607,312 | ✅ |
| soccernetv2 | 6,327,223 | ✅ |
| soccernet_ball | 1,306,500 | ✅ |
| tennis | 1,305,318 | ✅ |
| TouchMoment | 1,046,066 | ✅ (= HOI4D 855,000 + TACO 191,066) |
| fs_perf | 728,775 | ✅ |
| finediving | 312,256 | ✅ |
| fs_comp | **0** | ❌ annotations exist, **no frames extracted** |
| **total** | **18,633,450** (17,120 clips) | |

### Train-split frames (Σ `num_frames` in each `train.json`; verified == actual jpg where frames exist)
| dataset / subpart | train clips | train frames | trainable? |
|---|--:|--:|:--:|
| finegym | 3,327 | 4,686,067 | ✅ |
| soccernetv2 | 600 | 3,439,393 | ✅ |
| TouchMoment/HOI4D | 2,288 | 686,400 | ✅ |
| soccernet_ball | 4 | 576,619 | ✅ |
| tennis | 1,368 | 483,071 | ✅ |
| fs_perf | 79 | 336,325 | ✅ |
| finediving | 1,801 | 187,591 | ✅ |
| TouchMoment/TACO | 946 | 156,053 | ✅ |
| **fs_comp** | 178 | 759,400 | ❌ **no frames on PVC** |
| **TouchMoment/native** | 3,234 | 842,453 | ❌ **no frames on PVC** |

**Trainable "everything except HOI4D" total ≈ 9,865,119 train frames** (the 8 ✅ rows minus HOI4D).
Do **not** use the naive "all train.json" sum of 11,466,972 — it counts fs_comp + native-TouchMoment frames that don't exist.

---

## 6. Annotation format & split→frames mapping

Every `*.json` split file is a **list** of clips, unified schema:
```json
{"video": "H2_C5_N23_S263_s04_T4", "num_frames": 300, "num_events": 2,
 "events": [{"frame": 57, "label": "touch", "comment": "auto"}, ...],
 "fps": 15, "width": 854, "height": 480}
```
- `video` = the clip name. **Index clip id = `f"{dataset}/{video}"`** (e.g. HOI4D `video="H2_..."` → index clip `"TouchMoment/H2_..."`).
- `num_frames` = frame count for that clip (equals the actual jpg count where frames exist).
- `events` = spotting labels: `frame` index + `label`. This is the supervision target.
- TouchMoment sub-splits live under `TouchMoment/Annotations/{HOI4D,TACO,TouchMoment}/{train,val,test}.json`; all other datasets have `train/val/test[/challenge].json` at their top level.

To build a training list: read `train.json`, map each `video` → clip id `f"{ds}/{video}"`, pull frames via `ClipDataset`/`TarStore`, and use `events[].frame` as labels. Skip clips whose clip id is absent from `TarStore.clips()` (relevant only for the non-trainable datasets in §5).

---

## 7. Running a training Job on DPC

Key requirements: `kind: Job`, `restartPolicy: Never`, `backoffLimit: 0`, `activeDeadlineSeconds`, `imagePullSecrets`, mount the PVC at `/data`, a `Memory` `emptyDir` at `/dev/shm` (prevents dataloader OOM), `GITLAB_TOKEN` from `gitlab-token`, and `set -euo pipefail` in the command. **`nvidia.com/gpu` ≤ 4.**

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: vlm-spotting-train
spec:
  backoffLimit: 0
  activeDeadlineSeconds: 86400          # set to expected max runtime
  template:
    spec:
      restartPolicy: Never
      imagePullSecrets:
        - name: gitlab-docker-secret
      volumes:
        - name: shared-memory
          emptyDir: { medium: Memory }
        - name: dataset-volume
          persistentVolumeClaim:
            claimName: action-spotting-pvc
      containers:
        - name: train
          image: docker.aiml.team/<username>/<project>:latest   # CUDA 12.x, has torch+Pillow+as_dataset.py
          imagePullPolicy: Always
          command: ["/bin/bash", "-c"]
          args:
            - |
              set -euo pipefail
              mkdir -p /data/runs
              git clone "https://<username>:${GITLAB_TOKEN}@gitlab.aiml.team/<username>/<repo>.git"
              cd <repo>
              python train.py \
                --data-root /data/Action_Spotting \
                --index /data/Action_Spotting/index.sqlite \
                2>&1 | tee /data/runs/train_$(date +%Y%m%d_%H%M%S).log
          resources:
            limits:
              nvidia.com/gpu: 4          # 1, 2, or 4 — NEVER >4 (Job would be suspended)
          volumeMounts:
            - { name: dataset-volume, mountPath: /data }
            - { name: shared-memory,  mountPath: /dev/shm }
          env:
            - name: GITLAB_TOKEN
              valueFrom: { secretKeyRef: { name: gitlab-token, key: access-token } }
```
```bash
kubectl create -f train-job.yaml
kubectl logs -f job/vlm-spotting-train      # follow
kubectl get pods --selector=job-name=vlm-spotting-train
kubectl delete job vlm-spotting-train       # REQUIRED cleanup when done
```

**Image note:** bake `as_dataset.py` (+ Pillow, torch) into the training image, or `pip install pillow` and copy the file in. The dataset reader has no other deps (stdlib `sqlite3`).

### Multi-GPU / multi-node
- ≤4 A100 in one Job: `torchrun --nproc_per_node=4 …`, NCCL uses NVLink automatically.
- \>4 GPUs: launch multiple Jobs/pods (each ≤4 GPU) and run PyTorch DDP across them over InfiniBand (`mlx5`). Coordinate rendezvous (e.g. a headless Service / static master addr). Ask admins if you need this often.

---

## 8. Files shipped / tooling (on this machine, `/home/chang/DPC_instruction/`)

| file | purpose |
|---|---|
| `as_dataset.py` | **the reader** — `TarStore` (random file read) + `ClipDataset` (torch). Copied next to this doc; put it in the training image/repo. |
| `build_index_parallel.py` | rebuild `index.sqlite` if frame tars ever change (16-way, ~8 min). You should not need it. |
| `pack_and_upload.sh`, `dpc-pvc.yaml`, `dpc-holder-job.yaml` | how the data was uploaded (tar-by-clip shards → PVC via a holder pod). Reference only. |

### If you need to inspect/modify data on the PVC
Spin up a holder pod (mounts the PVC at `/data`), `kubectl exec` in, then delete it:
```bash
kubectl create -f /home/chang/DPC_instruction/dpc-holder-job.yaml
POD=$(kubectl get pods --selector=job-name=action-spotting-upload -o jsonpath='{.items[0].metadata.name}')
kubectl exec -it $POD -- bash            # /data/Action_Spotting is here
kubectl delete job action-spotting-upload   # holder holds 2CPU+8Gi until deleted — delete it!
```

---

## 9. Shared-resource rules (do not violate)

- Never hold a GPU idle; request only the GPUs you use (≤4). Comment out `nvidia.com/gpu` for non-GPU jobs.
- Always `backoffLimit: 0` + `activeDeadlineSeconds`.
- Delete Jobs (and any `sleep infinity` holder pods) as soon as they're done.
- PVC storage is temporary (~180 days) — the raw dataset is backed up elsewhere; treat `/data` as working storage.
- Long-running large-GPU workloads: arrange with `admins@aiml.team` first.
