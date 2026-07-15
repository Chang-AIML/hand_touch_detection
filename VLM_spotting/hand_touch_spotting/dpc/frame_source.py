"""Frame sources for the windowed spotting dataset (dpc/windowed_dataset.py).

Two interchangeable backends expose the SAME small API so training / feature-
extraction code stays storage-agnostic:

  * TarFrameSource(root, index_sqlite) — PRODUCTION on DPC. Wraps
    ``as_dataset.TarStore`` (the tar-sharded store: index.sqlite + shard files,
    fork-safe lazy handles). Frames are read by ``seek(offset)`` straight out of
    the shard, no unpacking.
  * DirFrameSource(root) — LOCAL testing. Loose ``*.jpg`` frames on disk at
        <root>/<dataset>/<framesdir>/<clipname>/*.jpg
    with framesdir == 'Frames' for dataset 'TouchMoment', else 'frames'.

Shared API (both classes):
    list_frames(video_id)            -> [frame_key, ...]            # natural-sorted
    load_window(video_id, start, end) -> np.uint8 (T, H, W, 3) RGB  # frames [start, end)

``video_id`` is '<dataset>/<clipname>' — exactly the clip id emitted by
dpc/make_dpc_annotations.py and used as the clip id in
/data/Action_Spotting/index.sqlite (see as_dataset.TarStore.clips()).

To switch a training run from local loose frames to the DPC tar store, swap the
constructor only — the WindowedSpottingDataset code is unchanged:

    # local
    fs = DirFrameSource("/home/chang/Dataset/Action_Spotting")
    # DPC (inside the training image / on the PVC)
    fs = TarFrameSource("/data/Action_Spotting",
                        "/data/Action_Spotting/index.sqlite")
"""
from __future__ import annotations

import io
import os
import re
import sys
import threading
from glob import glob

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _natural_key(s: str):
    """Natural sort key: split digit / non-digit runs so '000009.jpg' < '000010.jpg'
    and 'frame2' < 'frame10'. Frame files here are zero-padded so lexical order
    already works, but this is robust to any naming."""
    base = os.path.basename(s)
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r"(\d+)", base)]


def _decode_stack(jpeg_bytes_list) -> np.ndarray:
    """Decode a list of JPEG byte-strings to a contiguous np.uint8 (T,H,W,3) RGB."""
    imgs = [np.asarray(Image.open(io.BytesIO(b)).convert("RGB")) for b in jpeg_bytes_list]
    if not imgs:
        return np.zeros((0, 0, 0, 3), dtype=np.uint8)
    return np.ascontiguousarray(np.stack(imgs, axis=0)).astype(np.uint8)


def _clamp_window(n_frames: int, start: int, end):
    """Clamp a requested [start, end) span to the available frame count."""
    if end is None:
        end = n_frames
    start = max(0, int(start))
    end = min(int(end), n_frames)
    if end < start:
        end = start
    return start, end


def _import_tarstore():
    """Import as_dataset.TarStore (it lives one level above the HTS repo root:
    /home/chang/Project/VLM_spotting/as_dataset.py)."""
    try:
        from as_dataset import TarStore  # noqa: WPS433
        return TarStore
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))          # HTS/dpc
        parent_of_repo = os.path.dirname(os.path.dirname(here))    # -> VLM_spotting
        if parent_of_repo not in sys.path:
            sys.path.insert(0, parent_of_repo)
        from as_dataset import TarStore  # noqa: WPS433
        return TarStore


# --------------------------------------------------------------------------- #
# base
# --------------------------------------------------------------------------- #
class FrameSource:
    """Abstract frame source. Subclasses implement ``_frames`` (ordered per-clip
    frame records) and ``_read`` (record -> jpeg bytes)."""

    def _frames(self, video_id):
        """Return the natural-sorted list of per-frame records for this clip."""
        raise NotImplementedError

    def _read(self, record) -> bytes:
        """Return the raw JPEG bytes for one frame record from ``_frames``."""
        raise NotImplementedError

    def list_frames(self, video_id):
        """Ordered list of frame KEYS (paths) for this clip."""
        return [self._key(r) for r in self._frames(video_id)]

    @staticmethod
    def _key(record):
        return record

    def load_window(self, video_id, start, end) -> np.ndarray:
        """Decode frames [start, end) -> np.uint8 (T,H,W,3) RGB. The span is
        clamped to the frames actually available, so an over-long final window
        yields a shorter T rather than an error."""
        frames = self._frames(video_id)
        start, end = _clamp_window(len(frames), start, end)
        return _decode_stack([self._read(r) for r in frames[start:end]])


# --------------------------------------------------------------------------- #
# tar-sharded store (DPC / production)
# --------------------------------------------------------------------------- #
class TarFrameSource(FrameSource):
    """Frame source backed by as_dataset.TarStore (tar shards + index.sqlite)."""

    def __init__(self, root: str, index_sqlite: str):
        self.root = root
        self.index_sqlite = index_sqlite
        self._TarStore = _import_tarstore()
        self._local = threading.local()    # PER-THREAD store: sqlite conn + tar handles are NOT
                                           # thread-safe, so each prefetch thread gets its own.
        self._cache = {}                   # video_id -> rows (shared; writes idempotent/GIL-atomic)

    def _store_lazy(self):
        # TarStore opens its sqlite connection / shard handles lazily on first use. One store per
        # THREAD (thread-local) so ThreadPool prefetch can decode concurrently without racing the
        # sqlite connection / shard file offsets. Still fork-safe (each process/thread builds its own).
        st = getattr(self._local, "store", None)
        if st is None:
            st = self._local.store = self._TarStore(self.root, self.index_sqlite)
        return st

    def _frames(self, video_id):
        rows = self._cache.get(video_id)
        if rows is None:
            # TarStore.clip_frames already returns rows ordered by `ord`; re-sort by
            # the natural key of the path to honour the "sort by filename" contract.
            rows = sorted(self._store_lazy().clip_frames(video_id),
                          key=lambda r: _natural_key(r[0]))
            if not rows:
                raise KeyError(f"no frames for clip {video_id!r} in {self.index_sqlite}")
            self._cache[video_id] = rows
        return rows

    @staticmethod
    def _key(record):
        return record[0]                   # the tar path, e.g. 'finegym/clip/000076.jpg'

    def _read(self, record) -> bytes:
        _path, shard, off, size = record
        return self._store_lazy()._read_at(shard, off, size)


# --------------------------------------------------------------------------- #
# loose-jpg directory (local testing)
# --------------------------------------------------------------------------- #
class DirFrameSource(FrameSource):
    """Frame source backed by loose ``*.jpg`` files on disk.

    Layout: <root>/<dataset>/<framesdir>/<clipname>/*.jpg with
    framesdir == 'Frames' for dataset 'TouchMoment', else 'frames'.
    video_id == '<dataset>/<clipname>'.
    """

    def __init__(self, root: str):
        self.root = root
        self._cache = {}                   # video_id -> [abs_path, ...]

    def _clip_dir(self, video_id: str) -> str:
        if "/" not in video_id:
            raise ValueError(f"video_id must be '<dataset>/<clipname>', got {video_id!r}")
        dataset, clipname = video_id.split("/", 1)
        framesdir = "Frames" if dataset == "TouchMoment" else "frames"
        return os.path.join(self.root, dataset, framesdir, clipname)

    def _frames(self, video_id):
        paths = self._cache.get(video_id)
        if paths is None:
            d = self._clip_dir(video_id)
            paths = sorted(glob(os.path.join(d, "*.jpg")), key=_natural_key)
            if not paths:
                raise KeyError(f"no *.jpg frames for clip {video_id!r} under {d}")
            self._cache[video_id] = paths
        return paths

    def _read(self, record) -> bytes:
        with open(record, "rb") as fh:
            return fh.read()
