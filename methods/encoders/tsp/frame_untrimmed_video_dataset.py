from __future__ import division, print_function

import os
import glob
import pandas as pd
import numpy as np
import torch
import h5py

from torch.utils.data import Dataset
from PIL import Image


class FrameUntrimmedVideoDataset(Dataset):
    '''
    Drop-in replacement for UntrimmedVideoDataset that reads pre-extracted JPG
    frames instead of decoding video files (no ffmpeg / mp4 needed).

    Frames are expected at:  <root_dir>/<video_name>/{:06d}.jpg
    where <video_name> = basename(filename) without extension. The CSV is the
    same as TSP's tsp_groundtruth CSV (filename, fps, t-start, t-end,
    <label columns>, video-duration).
    '''

    IMG_NAME = '{:06d}.jpg'

    def __init__(self, csv_filename, root_dir, clip_length, frame_rate, clips_per_segment, temporal_jittering,
                 label_columns, label_mappings, seed=42, transforms=None, global_video_features=None, debug=False):
        df = FrameUntrimmedVideoDataset._clean_df_and_remove_short_segments(
            pd.read_csv(csv_filename), clip_length, frame_rate)
        self.df = FrameUntrimmedVideoDataset._check_frame_dirs_exist(df, root_dir)
        self.root_dir = root_dir
        self.clip_length = clip_length
        self.frame_rate = frame_rate
        self.clips_per_segment = clips_per_segment

        self.temporal_jittering = temporal_jittering
        self.rng = np.random.RandomState(seed=seed)
        self.uniform_sampling = np.linspace(0, 1, clips_per_segment)

        self.transforms = transforms

        self.label_columns = label_columns
        self.label_mappings = label_mappings
        for label_column, label_mapping in zip(label_columns, label_mappings):
            self.df[label_column] = self.df[label_column].map(lambda x: -1 if pd.isnull(x) else label_mapping[x])

        # cache number of frames per (unique) video to clamp indices
        self._num_frames = {}
        for name in self.df['video-name'].unique():
            d = os.path.join(root_dir, name)
            self._num_frames[name] = len(glob.glob(os.path.join(d, '*.jpg')))

        self.global_video_features = global_video_features
        self.debug = debug

    def __len__(self):
        return len(self.df) * self.clips_per_segment if not self.debug else 100

    def __getitem__(self, idx):
        sample = {}
        row = self.df.iloc[idx % len(self.df)]
        video_name, fps, t_start, t_end = row['video-name'], row['fps'], row['t-start'], row['t-end']

        # NATIVE-FRAME sampling: clip = clip_length CONSECUTIVE native frames of this
        # video (step=1, no fps resampling). Mixed-fps datasets just work -- a 15fps
        # clip spans 0.8s, a 30fps clip 0.4s, but the model only sees clip_length frames.
        clip_length_in_sec = self.clip_length / fps
        ratio = self.rng.uniform() if self.temporal_jittering else self.uniform_sampling[idx // len(self.df)]
        clip_t_start = t_start + ratio * (t_end - t_start - clip_length_in_sec)

        n = self._num_frames[video_name]
        start_f = int(round(clip_t_start * fps))
        frame_idxs = [min(max(start_f + k, 0), n - 1) for k in range(self.clip_length)]

        d = os.path.join(self.root_dir, video_name)
        frames = [np.asarray(Image.open(os.path.join(d, self.IMG_NAME.format(j))).convert('RGB'))
                  for j in frame_idxs]
        vframes = torch.from_numpy(np.stack(frames))   # [T, H, W, C] uint8

        if vframes.shape[0] != self.clip_length:
            raise RuntimeError(f'<FrameUntrimmedVideoDataset>: got clip of length {vframes.shape[0]} '
                               f'!= {self.clip_length}. video={video_name}, clip_t_start={clip_t_start}, fps={fps}')

        sample['clip'] = self.transforms(vframes)
        for label_column in self.label_columns:
            sample[label_column] = row[label_column]

        if self.global_video_features:
            f = h5py.File(self.global_video_features, 'r')
            sample['gvf'] = torch.tensor(f[video_name][()])
            f.close()

        return sample

    @staticmethod
    def _clean_df_and_remove_short_segments(df, clip_length, frame_rate):
        df['t-end'] = np.minimum(df['t-end'], df['video-duration'])
        df['t-start'] = np.maximum(df['t-start'], 0)

        # NATIVE frames: a segment is long enough if it spans >= clip_length of its
        # OWN frames (per-row fps), so mixed-fps videos are judged consistently.
        segment_length = (df['t-end'] - df['t-start']) * df['fps']
        mask = segment_length >= clip_length
        num_segments = len(df)
        num_segments_to_keep = sum(mask)
        if num_segments - num_segments_to_keep > 0:
            df = df[mask].reset_index(drop=True)
            print(f'<FrameUntrimmedVideoDataset>: removed {num_segments - num_segments_to_keep}='
                  f'{100 * (1 - num_segments_to_keep / num_segments):.2f}% from the {num_segments} '
                  f'segments shorter than clip_length={clip_length} native frames.')
        return df

    @staticmethod
    def _check_frame_dirs_exist(df, root_dir):
        # video-name = filename without extension (matches GVF h5 keys)
        df['video-name'] = df['filename'].map(lambda f: os.path.splitext(os.path.basename(f))[0])
        for name in df.drop_duplicates('video-name')['video-name'].values:
            d = os.path.join(root_dir, name)
            if not os.path.isdir(d):
                raise ValueError(f'<FrameUntrimmedVideoDataset>: frame dir={d} does not exist. '
                                 f'Double-check root_dir (--frames-dir) and csv_filename.')
        return df
