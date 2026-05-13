"""
feeders/feeder_ntu_binary.py
============================
DataLoader untuk binary fall detection berbasis skeleton NTU RGB+D 25-joint.

Return dari __getitem__: (data tensor (C,T,V,M), label int)
  - data : shape (3, window_size, 25, 1)  dtype float32
    channel 0 = x (world meter, relatif ke SpineBase)
    channel 1 = y
    channel 2 = z
  - label: 0 = not_fall, 1 = fall
"""

import pickle
import random
from collections import Counter
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

NUM_JOINTS = 25
CHANNELS   = 3

# Pasangan kiri-kanan untuk flip horizontal (0-indexed)
# NTU joint names:
#  4=ShoulderLeft  8=ShoulderRight
#  5=ElbowLeft     9=ElbowRight
#  6=WristLeft    10=WristRight
#  7=HandLeft     11=HandRight
# 12=HipLeft      16=HipRight
# 13=KneeLeft     17=KneeRight
# 14=AnkleLeft    18=AnkleRight
# 15=FootLeft     19=FootRight
# 21=HandTipLeft  23=HandTipRight
# 22=ThumbLeft    24=ThumbRight
FLIP_PAIRS = [
    (4, 8), (5, 9), (6, 10), (7, 11),
    (12, 16), (13, 17), (14, 18), (15, 19),
    (21, 23), (22, 24),
]


class Feeder(Dataset):
    def __init__(
        self,
        data_path,
        label_path,
        split         = "train",
        window_size   = 64,
        p_interval    = None,
        random_move   = False,   # random 3D rotation (penting untuk NTU)
        random_shift  = False,   # random global translation
        random_flip   = False,   # horizontal flip
        random_speed  = False,   # temporal speed perturbation
        random_noise  = False,   # Gaussian noise pada koordinat
        normalization = False,   # min-max normalisasi ke [-1,1]
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
            f"Ekspektasi V={NUM_JOINTS} (NTU 25 joints), dapat V={V}"
        assert M == 1, \
            f"Ekspektasi M=1 (single person per sampel), dapat M={M}"
        assert C == CHANNELS, \
            f"Ekspektasi C={CHANNELS} (x,y,z), dapat C={C}"

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
        if self.do_move:  x = self._rotate3d(x)
        if self.do_flip:  x = self._flip(x)
        if self.do_speed: x = self._speed_perturb(x)
        if self.do_noise: x = self._add_noise(x)
        if self.do_norm:  x = self._normalize(x)

        return torch.tensor(x, dtype=torch.float32), y

    # ── Valid frame counting ───────────────────────────────────────────────────

    def _count_valid_frames(self, x):
        """
        Hitung frame yang punya data (bukan semua-nol).
        NTU: frame kosong = semua koordinat nol karena padding.
        """
        spatial = x[:, :, :, 0]          # (C, T, V)
        valid = int((spatial != 0).any(axis=(0, 2)).sum())
        return max(valid, 1)

    # ── Temporal crop + interpolasi ke window_size ─────────────────────────────

    def _temporal_crop(self, x, valid, length):
        T      = x.shape[1]
        length = min(length, valid, T)
        max_start = max(0, valid - length)

        if self.is_train:
            start = random.randint(0, max_start)
        else:
            start = max_start // 2

        seg = x[:, start: start + length, :, :]

        if seg.shape[1] == self.window_size:
            return seg

        idx = np.linspace(0, seg.shape[1] - 1, self.window_size, dtype=int)
        return seg[:, idx, :, :]

    # ── Augmentasi ─────────────────────────────────────────────────────────────

    def _shift(self, x):
        """Random global translation (translasi dalam meter, misal ±0.1m)."""
        x = x.copy()
        x[0] += random.uniform(-0.1, 0.1)  # x
        x[1] += random.uniform(-0.1, 0.1)  # y
        x[2] += random.uniform(-0.1, 0.1)  # z
        return x

    def _rotate3d(self, x):
        """
        Random rotasi 3D di sekitar sumbu Y (vertikal, orang berdiri).
        Tambahan rotasi kecil pada sumbu X dan Z untuk variasi.
        """
        x   = x.copy()
        # Rotasi utama: sumbu Y (yaw) ±30 derajat
        yaw   = random.uniform(-0.52, 0.52)   # ±30°
        # Rotasi minor: pitch & roll ±10°
        pitch = random.uniform(-0.17, 0.17)
        roll  = random.uniform(-0.17, 0.17)

        # Matriks rotasi Ry (yaw, sumbu Y)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
        Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
        R  = Ry @ Rx @ Rz   # (3, 3)

        # x shape: (C=3, T, V=25, M=1)
        coords = x[:, :, :, 0]        # (3, T, 25)
        coords = np.einsum('ij,jkl->ikl', R, coords)  # (3, T, 25)
        x[:, :, :, 0] = coords
        return x

    def _flip(self, x):
        """
        Flip horizontal (sumbu X) dengan swap pasangan joint kiri-kanan.
        50% probabilitas.
        """
        if random.random() > 0.5:
            return x
        x = x.copy()
        x[0] = -x[0]   # balik koordinat X
        for l_idx, r_idx in FLIP_PAIRS:
            x[:, :, [l_idx, r_idx], :] = x[:, :, [r_idx, l_idx], :]
        return x

    def _speed_perturb(self, x):
        """Random temporal resampling (0.75× – 1.25× kecepatan)."""
        C, T, V, M = x.shape
        factor  = random.uniform(0.75, 1.25)
        new_len = max(1, int(T * factor))
        src_idx = np.linspace(0, T - 1, new_len, dtype=int)
        tgt_idx = np.linspace(0, new_len - 1, T, dtype=int)
        return x[:, src_idx, :, :][:, tgt_idx, :, :]

    def _add_noise(self, x):
        """Gaussian noise kecil pada koordinat (sigma=0.005m ≈ 5mm)."""
        x = x.copy()
        noise = np.random.normal(0, 0.005, x.shape).astype(np.float32)
        x += noise
        return x

    def _normalize(self, x):
        """Min-max normalisasi setiap channel ke [-1, 1]."""
        x = x.copy()
        for c in range(CHANNELS):
            mn, mx = x[c].min(), x[c].max()
            if mx - mn > 1e-6:
                x[c] = 2.0 * (x[c] - mn) / (mx - mn) - 1.0
        return x

    # ── Utilitas ──────────────────────────────────────────────────────────────

    def top_k(self, score, top_k):
        """Top-k accuracy. score: (N, num_class) numpy array."""
        rank = score.argsort()[:, ::-1]
        hit  = [l in rank[i, :top_k] for i, l in enumerate(self.label)]
        return sum(hit) / len(hit)

    def get_weighted_sampler(self):
        """WeightedRandomSampler untuk balanced batch sampling."""
        cnt = Counter(self.label)
        sw  = [1.0 / cnt[l] for l in self.label]
        return WeightedRandomSampler(sw, len(sw), replacement=True)

    def class_distribution(self):
        c = Counter(self.label)
        return {"not_fall": c[0], "fall": c[1], "total": len(self.label)}
