"""
feeders/fall_feeder.py
=======================
DataLoader binary fall detection.
Letakkan di: BlockGCN/feeders/fall_feeder.py

Mendukung format pkl: (sample_names, labels)
  - sample_names: list nama file (stem), dipakai test script untuk output .txt
  - labels: list int 0/1
"""

import pickle
import random
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

NUM_JOINTS = 17
CHANNELS   = 3
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
        self.is_train    = split == "train"

        self.do_move  = random_move  and self.is_train
        self.do_shift = random_shift and self.is_train
        self.do_flip  = random_flip  and self.is_train
        self.do_speed = random_speed and self.is_train
        self.do_norm  = normalization

        with open(label_path, "rb") as f:
            self.sample_name, self.label = pickle.load(f)

        # sample_name bisa berupa list nama file ATAU None (backward compat)
        if self.sample_name is None:
            self.sample_name = [str(i) for i in range(len(self.label))]

        self.data = (np.load(data_path, mmap_mode="r")
                     if use_mmap else np.load(data_path))

        if debug:
            self.data        = self.data[:100]
            self.label       = self.label[:100]
            self.sample_name = self.sample_name[:100]

        N, C, T, V, M = self.data.shape
        assert V == 17, f"Ekspektasi V=17 (COCO joints), dapat V={V}"
        assert M == 1,  f"Ekspektasi M=1 (1 orang), dapat M={M}"
        assert C == 3,  f"Ekspektasi C=3 (x,y,conf), dapat C={C}"

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        x = np.array(self.data[idx], np.float32)  # (C, T, V, 1)
        y = int(self.label[idx])

        valid = self._valid_frames(x)
        p     = random.uniform(*self.p_interval) if self.is_train else self.p_interval[-1]
        crop  = max(1, int(valid * p))
        x     = self._crop(x, valid, crop)

        if self.do_shift: x = self._shift(x)
        if self.do_move:  x = self._rotate_scale(x)
        if self.do_flip:  x = self._flip(x)
        if self.do_speed: x = self._speed(x)
        if self.do_norm:  x = self._normalize(x)

        return torch.tensor(x, dtype=torch.float32), y

    # ── Augmentasi ────────────────────────────────────────────────────────────

    def _valid_frames(self, x):
        return max(int((x[2, :, :, 0] > 0).any(axis=1).sum()), 1)

    def _crop(self, x, valid, length):
        if self.is_train:
            max_s = max(0, valid - length)
            start = random.randint(max_s // 2, max_s) if (random.random() < 0.7 and max_s > 0) \
                    else random.randint(0, max_s)
        else:
            start = max(0, (valid - length) // 2)

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
        x = x.copy()
        th = random.uniform(-0.25, 0.25)
        sc = random.uniform(0.9, 1.1)
        c, s = np.cos(th), np.sin(th)
        x0, x1 = x[0].copy(), x[1].copy()
        x[0] = sc * (c*x0 - s*x1)
        x[1] = sc * (s*x0 + c*x1)
        return x

    def _flip(self, x):
        if random.random() > 0.5:
            return x
        x = x.copy()
        x[0] = -x[0]
        for l, r in FLIP_PAIRS:
            x[:, :, [l, r], :] = x[:, :, [r, l], :]
        return x

    def _speed(self, x):
        C, T, V, M = x.shape
        fac = random.uniform(0.8, 1.2)
        nl  = max(1, int(T * fac))
        src = np.linspace(0, T-1, nl, dtype=int)
        tgt = np.linspace(0, nl-1, T, dtype=int)
        return x[:, src, :, :][:, tgt, :, :]

    def _normalize(self, x):
        x = x.copy()
        for c in range(2):
            mn, mx = x[c].min(), x[c].max()
            if mx - mn > 1e-6:
                x[c] = 2*(x[c]-mn)/(mx-mn) - 1
        return x

    # ── Utilitas ──────────────────────────────────────────────────────────────

    def top_k(self, score: np.ndarray, top_k: int) -> float:
        """Dipanggil oleh processor BlockGCN untuk eval."""
        rank = score.argsort()[:, ::-1]
        hit  = [l in rank[i, :top_k] for i, l in enumerate(self.label)]
        return sum(hit) / len(hit)

    def get_weighted_sampler(self):
        cnt = Counter(self.label)
        sw  = [1.0/cnt[l] for l in self.label]
        return WeightedRandomSampler(sw, len(sw), replacement=True)

    def class_distribution(self):
        c = Counter(self.label)
        return {"not_fall": c[0], "fall": c[1], "total": len(self.label)}