"""
feeders/feeder_yolo.py
=======================
DataLoader untuk binary fall detection berbasis skeleton YOLO11-pose.

Return dari __getitem__: (data tensor (C,T,V,M), label int)
  - data : shape (3, window_size, 17, 1)  dtype float32
  - label: 0 = not_fall, 1 = fall

Format pkl yang didukung: (sample_names, labels)
"""

import pickle
import random
from collections import Counter
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

NUM_JOINTS = 17
CHANNELS   = 3

# Pasangan joint kiri-kanan untuk flip augmentasi (COCO format)
FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


class Feeder(Dataset):
    def __init__(
        self,
        data_path,
        label_path,
        split         = "train",
        window_size   = 150,
        p_interval    = None,
        random_move   = False,
        random_shift  = False,
        random_flip   = False,
        random_speed  = False,
        random_noise  = False,
        normalization = False,
        debug         = False,
        use_mmap      = True,
    ):
        self.split       = split
        self.window_size = window_size
        self.p_interval  = p_interval if p_interval is not None else [1.0, 1.0]
        self.is_train    = (split == "train")

        self.do_move  = random_move  and self.is_train
        self.do_shift = random_shift and self.is_train
        self.do_flip  = random_flip  and self.is_train
        self.do_speed = random_speed and self.is_train
        self.do_noise = random_noise and self.is_train
        self.do_norm  = normalization

        with open(label_path, "rb") as f:
            self.sample_name, self.label = pickle.load(f)

        if self.sample_name is None:
            self.sample_name = [str(i) for i in range(len(self.label))]

        if use_mmap:
            self.data = np.load(data_path, mmap_mode="r")
        else:
            self.data = np.load(data_path)

        if debug:
            self.data        = self.data[:100]
            self.label       = self.label[:100]
            self.sample_name = self.sample_name[:100]

        N, C, T, V, M = self.data.shape
        assert V == NUM_JOINTS, \
            "Ekspektasi V={} (COCO joints), dapat V={}".format(NUM_JOINTS, V)
        assert M == 1, \
            "Ekspektasi M=1 (single person), dapat M={}".format(M)
        assert C == CHANNELS, \
            "Ekspektasi C={} (x,y,conf), dapat C={}".format(CHANNELS, C)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        x = np.array(self.data[idx], dtype=np.float32)  # (C, T, V, M)
        y = int(self.label[idx])

        valid = self._count_valid_frames(x)
        p     = (random.uniform(self.p_interval[0], self.p_interval[1])
                 if self.is_train else self.p_interval[-1])
        crop  = max(1, int(valid * p))
        x     = self._temporal_crop(x, valid, crop)

        if self.do_shift: x = self._shift(x)
        if self.do_move:  x = self._rotate_scale(x)
        if self.do_flip:  x = self._flip(x)
        if self.do_speed: x = self._speed_perturb(x)
        if self.do_noise: x = self._add_noise(x)
        if self.do_norm:  x = self._normalize(x)

        return torch.tensor(x, dtype=torch.float32), y

    # ── Valid frame counting ───────────────────────────────────────────────────

    def _count_valid_frames(self, x):
        """Count frames where at least one joint has nonzero confidence."""
        conf  = x[2, :, :, 0]                 # (T, V)
        valid = int((conf > 0).any(axis=1).sum())
        return max(valid, 1)

    # ── Temporal crop + interpolation to window_size ───────────────────────────

    def _temporal_crop(self, x, valid, length):
        """
        Randomly (train) or deterministically (val) select `length` consecutive
        frames from the valid portion, then interpolate to window_size.
        """
        T        = x.shape[1]
        length   = min(length, valid, T)
        max_start = max(0, valid - length)

        if self.is_train:
            start = random.randint(0, max_start)
        else:
            start = max_start // 2

        seg = x[:, start: start + length, :, :]   # (C, length, V, M)

        if seg.shape[1] == self.window_size:
            return seg

        # Temporal interpolation: linearly sample window_size indices from seg
        idx = np.linspace(0, seg.shape[1] - 1, self.window_size, dtype=int)
        return seg[:, idx, :, :]

    # ── Augmentation ──────────────────────────────────────────────────────────

    def _shift(self, x):
        """Random global translation of x,y coordinates."""
        x = x.copy()
        x[0] += random.uniform(-0.15, 0.15)
        x[1] += random.uniform(-0.15, 0.15)
        return x

    def _rotate_scale(self, x):
        """Random 2-D rotation (±20°) and scale (0.85–1.15)."""
        x  = x.copy()
        th = random.uniform(-0.35, 0.35)   # ~±20 degrees
        sc = random.uniform(0.85, 1.15)
        c, s    = np.cos(th), np.sin(th)
        x0, x1  = x[0].copy(), x[1].copy()
        x[0] = sc * (c * x0 - s * x1)
        x[1] = sc * (s * x0 + c * x1)
        return x

    def _flip(self, x):
        """Horizontal flip with COCO joint pair swap (50% probability)."""
        if random.random() > 0.5:
            return x
        x = x.copy()
        x[0] = -x[0]
        for l_idx, r_idx in FLIP_PAIRS:
            x[:, :, [l_idx, r_idx], :] = x[:, :, [r_idx, l_idx], :]
        return x

    def _speed_perturb(self, x):
        """Random temporal speed change (0.75×–1.25×) via resampling."""
        C, T, V, M = x.shape
        factor  = random.uniform(0.75, 1.25)
        new_len = max(1, int(T * factor))
        src_idx = np.linspace(0, T - 1, new_len, dtype=int)
        tgt_idx = np.linspace(0, new_len - 1, T, dtype=int)
        return x[:, src_idx, :, :][:, tgt_idx, :, :]

    def _add_noise(self, x):
        """Small Gaussian noise on x,y coordinates (sigma=0.01)."""
        x = x.copy()
        noise = np.random.normal(0, 0.01, x[:2].shape).astype(np.float32)
        x[:2] += noise
        return x

    def _normalize(self, x):
        """Min-max normalise x,y to [-1, 1]. Confidence channel unchanged."""
        x = x.copy()
        for c in range(2):
            mn, mx = x[c].min(), x[c].max()
            if mx - mn > 1e-6:
                x[c] = 2.0 * (x[c] - mn) / (mx - mn) - 1.0
        return x

    # ── Utilities ─────────────────────────────────────────────────────────────

    def top_k(self, score, top_k):
        """Compute top-k accuracy. score: (N, num_class) numpy array."""
        rank = score.argsort()[:, ::-1]
        hit  = [l in rank[i, :top_k] for i, l in enumerate(self.label)]
        return sum(hit) / len(hit)

    def get_weighted_sampler(self):
        """Return WeightedRandomSampler for balanced batch sampling."""
        cnt = Counter(self.label)
        sw  = [1.0 / cnt[l] for l in self.label]
        return WeightedRandomSampler(sw, len(sw), replacement=True)

    def class_distribution(self):
        c = Counter(self.label)
        return {"not_fall": c[0], "fall": c[1], "total": len(self.label)}
