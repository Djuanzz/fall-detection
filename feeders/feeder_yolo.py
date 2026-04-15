"""
feeders/fall_feeder.py
=======================
DataLoader binary fall detection.
Letakkan di: BlockGCN/feeders/fall_feeder.py

Return dari __getitem__: (data, label)
  - data : torch.Tensor shape (C, T, V, M) = (3, window_size, 17, 1)
  - label: int 0 atau 1

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
FLIP_PAIRS = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]


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
        normalization = False,
        debug         = False,
        use_mmap      = True,
    ):
        self.split       = split
        self.window_size = window_size
        self.p_interval  = p_interval or [1.0, 1.0]
        self.is_train    = (split == "train")

        self.do_move  = random_move  and self.is_train
        self.do_shift = random_shift and self.is_train
        self.do_flip  = random_flip  and self.is_train
        self.do_speed = random_speed and self.is_train
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
        """Return: (data tensor (C,T,V,M), label int)"""
        x = np.array(self.data[idx], dtype=np.float32)
        y = int(self.label[idx])

        valid = self._count_valid_frames(x)
        p     = random.uniform(self.p_interval[0], self.p_interval[1]) \
                if self.is_train else self.p_interval[-1]
        crop  = max(1, int(valid * p))
        x     = self._temporal_crop(x, valid, crop)

        if self.do_shift: x = self._shift(x)
        if self.do_move:  x = self._rotate_scale(x)
        if self.do_flip:  x = self._flip(x)
        if self.do_speed: x = self._speed_perturb(x)
        if self.do_norm:  x = self._normalize(x)

        return torch.tensor(x, dtype=torch.float32), y

    def _count_valid_frames(self, x):
        conf  = x[2, :, :, 0]
        valid = (conf > 0).any(axis=1).sum()
        return max(int(valid), 1)

    def _temporal_crop(self, x, valid, length):
        max_start = max(0, valid - length)
        if self.is_train:
            if random.random() < 0.7 and max_start > 0:
                start = random.randint(max_start // 2, max_start)
            else:
                start = random.randint(0, max_start)
        else:
            start = max_start // 2

        seg = x[:, start: start + length, :, :]
        if seg.shape[1] != self.window_size:
            idx = np.linspace(0, seg.shape[1] - 1, self.window_size, dtype=int)
            seg = seg[:, idx, :, :]
        return seg

    def _shift(self, x):
        x = x.copy()
        x[0] += random.uniform(-0.1, 0.1)
        x[1] += random.uniform(-0.1, 0.1)
        return x

    def _rotate_scale(self, x):
        x  = x.copy()
        th = random.uniform(-0.25, 0.25)
        sc = random.uniform(0.9, 1.1)
        c, s  = np.cos(th), np.sin(th)
        x0, x1 = x[0].copy(), x[1].copy()
        x[0] = sc * (c * x0 - s * x1)
        x[1] = sc * (s * x0 + c * x1)
        return x

    def _flip(self, x):
        if random.random() > 0.5:
            return x
        x = x.copy()
        x[0] = -x[0]
        for l_idx, r_idx in FLIP_PAIRS:
            x[:, :, [l_idx, r_idx], :] = x[:, :, [r_idx, l_idx], :]
        return x

    def _speed_perturb(self, x):
        C, T, V, M = x.shape
        factor  = random.uniform(0.8, 1.2)
        new_len = max(1, int(T * factor))
        src_idx = np.linspace(0, T - 1, new_len, dtype=int)
        tgt_idx = np.linspace(0, new_len - 1, T, dtype=int)
        return x[:, src_idx, :, :][:, tgt_idx, :, :]

    def _normalize(self, x):
        x = x.copy()
        for c in range(2):
            mn, mx = x[c].min(), x[c].max()
            if mx - mn > 1e-6:
                x[c] = 2 * (x[c] - mn) / (mx - mn) - 1
        return x

    def top_k(self, score, top_k):
        """Dipanggil Processor saat eval. score: (N, num_class) numpy array."""
        rank = score.argsort()[:, ::-1]
        hit  = [l in rank[i, :top_k] for i, l in enumerate(self.label)]
        return sum(hit) / len(hit)

    def get_weighted_sampler(self):
        cnt = Counter(self.label)
        sw  = [1.0 / cnt[l] for l in self.label]
        return WeightedRandomSampler(sw, len(sw), replacement=True)

    def class_distribution(self):
        c = Counter(self.label)
        return {"not_fall": c[0], "fall": c[1], "total": len(self.label)}