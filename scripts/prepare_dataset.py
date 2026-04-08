"""
step2_prepare_fall_dataset.py
==============================
Konversi skeleton .npy ke tensor BlockGCN-ready.
Membuat DUA versi sekaligus: balanced dan full.

PENTING: File .pkl menyimpan (sample_names, labels) — bukan (None, labels).
         sample_names dibutuhkan oleh test script untuk output .txt.

Output:
    data/fall_detection/
        balanced/joint/
            train_data.npy     (N, 3, T, 17, 1)
            train_label.pkl    (list_nama_file, list_label)
            val_data.npy
            val_label.pkl
            label_map.json
            dataset_info.json
        balanced/bone/  ...
        balanced/motion/ ...
        full/joint/  ...
        full/bone/   ...
        full/motion/ ...

Cara pakai:
    python step2_prepare_fall_dataset.py \
        --skeleton_dir skeleton_raw \
        --out_dir      data/fall_detection \
        --max_frames   150 \
        --val_split    0.2
"""

import argparse
import json
import pickle
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    sys.exit("pip install scikit-learn")

NUM_JOINTS = 17
CHANNELS   = 3
LABEL_MAP  = {"not_fall": 0, "fall": 1}


# ── Preprocessing ──────────────────────────────────────────────────────────────

def pad_or_crop(seq: np.ndarray, target: int) -> np.ndarray:
    """(T,V,C) → (target,V,C). Pad dengan nol atau crop dari tengah."""
    T = seq.shape[0]
    if T == target:
        return seq
    if T < target:
        pad = np.zeros((target - T, NUM_JOINTS, CHANNELS), np.float32)
        return np.concatenate([seq, pad], axis=0)
    start = (T - target) // 2
    return seq[start: start + target]


def to_tensor(seq: np.ndarray, target: int) -> np.ndarray:
    """(T,V,C) → (C,T,V,1)"""
    s = pad_or_crop(seq, target)
    return s.transpose(2, 0, 1)[:, :, :, np.newaxis]


def build_bone(seq: np.ndarray) -> np.ndarray:
    """Bone = selisih koordinat joint child - parent. (T,17,3)"""
    PAIRS = [
        (1,0),(2,0),(3,1),(4,2),
        (5,0),(6,0),(5,6),
        (7,5),(9,7),(8,6),(10,8),
        (11,5),(12,6),(11,12),
        (13,11),(15,13),(14,12),(16,14),
    ]
    b = np.zeros_like(seq)
    for c, p in PAIRS:
        b[:, c, :2] = seq[:, c, :2] - seq[:, p, :2]
        b[:, c, 2]  = np.minimum(seq[:, c, 2], seq[:, p, 2])
    return b


def build_motion(seq: np.ndarray) -> np.ndarray:
    """seq[t+1] - seq[t], frame terakhir = 0. (T,17,3)"""
    m = np.zeros_like(seq)
    m[:-1] = seq[1:] - seq[:-1]
    return m


# ── Collect ────────────────────────────────────────────────────────────────────

def collect_samples(skeleton_dir: Path) -> list:
    """
    Return list of (array, label_int, stem_name).
    stem_name = nama file tanpa ekstensi, misal 'S001C001P001R001A043'
    """
    samples = []
    for name, idx in LABEL_MAP.items():
        folder = skeleton_dir / name
        if not folder.exists():
            print(f"  [WARN] Folder tidak ada: {folder}")
            continue
        files = sorted(folder.glob("*.npy"))
        print(f"  {name:10s} (label={idx}): {len(files)} file")
        for fp in files:
            arr = np.load(str(fp))
            if arr.ndim == 3 and arr.shape[1] == NUM_JOINTS:
                samples.append((arr.astype(np.float32), idx, fp.stem))
            else:
                print(f"    [SKIP] shape salah {arr.shape}: {fp.name}")
    return samples


# ── Build & save ───────────────────────────────────────────────────────────────

def build_tensors(samples, max_frames, transform=None):
    N = len(samples)
    data   = np.zeros((N, CHANNELS, max_frames, NUM_JOINTS, 1), np.float32)
    labels = []
    names  = []
    for i, (arr, label, stem) in enumerate(samples):
        src = transform(arr) if transform else arr
        data[i] = to_tensor(src, max_frames)
        labels.append(label)
        names.append(stem)
    return data, labels, names


def save_split(out_dir, split, data, labels, names):
    np.save(str(out_dir / f"{split}_data.npy"), data)
    # Simpan (names, labels) — names dibutuhkan oleh test script
    with open(out_dir / f"{split}_label.pkl", "wb") as f:
        pickle.dump((names, labels), f)
    c = Counter(labels)
    print(f"    {split}_data.npy  shape={data.shape}  "
          f"not_fall={c[0]} fall={c[1]}")


def make_version(out_dir, train_s, val_s, max_frames, lmap_src,
                 version_name, extra_info=None, transform=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(lmap_src), str(out_dir / "label_map.json"))
    print(f"\n  [{version_name}]")

    td, tl, tn = build_tensors(train_s, max_frames, transform)
    vd, vl, vn = build_tensors(val_s,   max_frames, transform)
    save_split(out_dir, "train", td, tl, tn)
    save_split(out_dir, "val",   vd, vl, vn)

    tc = Counter(tl)
    vc = Counter(vl)
    pos_weight = round(tc[0] / max(tc[1], 1), 4)

    info = {
        "version":    version_name,
        "num_class":  2,
        "classes":    {"0": "not_fall", "1": "fall"},
        "num_joints": NUM_JOINTS,
        "channels":   CHANNELS,
        "max_frames": max_frames,
        "num_person": 1,
        "train": {"total": len(tl), "not_fall": tc[0], "fall": tc[1]},
        "val":   {"total": len(vl), "not_fall": vc[0], "fall": vc[1]},
        "pos_weight": pos_weight,
        "class_weight_for_config": [1.0, round(pos_weight, 2)],
    }
    if extra_info:
        info.update(extra_info)
    with open(out_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"    pos_weight={pos_weight} → class_weight: [1.0, {pos_weight}]")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeleton_dir", default="../dataset/ntu_skeleton")
    ap.add_argument("--out_dir",  default="../dataset/ntu_data")
    ap.add_argument("--max_frames", type=int, default=150)
    ap.add_argument("--val_split",  type=float, default=0.2)
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    skel = Path(args.skeleton_dir)
    out  = Path(args.out_dir)
    rng  = np.random.default_rng(args.seed)

    print(f"\nMembaca skeleton dari: {skel}")
    all_s = collect_samples(skel)
    if not all_s:
        sys.exit(f"Tidak ada sampel! Pastikan {skel}/fall/ dan {skel}/not_fall/ ada.")

    labels_all = [s[1] for s in all_s]
    cnt = Counter(labels_all)
    print(f"\nTotal: {len(all_s)}  (not_fall={cnt[0]}, fall={cnt[1]}, "
          f"rasio 1:{cnt[0]/max(cnt[1],1):.1f})")

    # Stratified split dari data PENUH dulu
    tr_idx, va_idx = train_test_split(
        range(len(all_s)), test_size=args.val_split,
        stratify=labels_all, random_state=args.seed)
    tr_all = [all_s[i] for i in tr_idx]
    va_all = [all_s[i] for i in va_idx]

    tc = Counter(s[1] for s in tr_all)
    print(f"Split → train={len(tr_all)} (not_fall={tc[0]}, fall={tc[1]}) "
          f"| val={len(va_all)}")

    lmap_src = skel / "label_map.json"

    # ── VERSI BALANCED ────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("Versi 1: BALANCED (undersample not_fall)")
    fall_tr     = [s for s in tr_all if s[1] == 1]
    not_fall_tr = [s for s in tr_all if s[1] == 0]
    idx_u = rng.choice(len(not_fall_tr), size=len(fall_tr), replace=False)
    not_fall_tr_bal = [not_fall_tr[i] for i in sorted(idx_u)]
    tr_bal = fall_tr + not_fall_tr_bal
    rng.shuffle(tr_bal)

    bal = out / "balanced"
    for mod, fn in [("joint", None), ("bone", build_bone), ("motion", build_motion)]:
        make_version(bal/mod, tr_bal, va_all, args.max_frames, lmap_src,
                     f"balanced/{mod}", transform=fn)

    # ── VERSI FULL ────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("Versi 2: FULL (semua data + class_weight)")
    full = out / "full"
    for mod, fn in [("joint", None), ("bone", build_bone), ("motion", build_motion)]:
        make_version(full/mod, tr_all, va_all, args.max_frames, lmap_src,
                     f"full/{mod}", transform=fn)

    # Tampilkan class weight yang harus dimasukkan ke config
    with open(full / "joint" / "dataset_info.json") as f:
        pw = json.load(f)["pos_weight"]

    print(f"\n{'='*50}")
    print("SELESAI.")
    print(f"\nUntuk config/fall_detection/full.yaml, set:")
    print(f"  loss_args:")
    print(f"    weight: [1.0, {pw}]")
    print(f"\nMulai training:")
    print(f"  python main.py --config config/fall_detection/balanced.yaml \\")
    print(f"                 --work-dir work_dir/fall_balanced --device 0")
    print(f"\n  python main.py --config config/fall_detection/full.yaml \\")
    print(f"                 --work-dir work_dir/fall_full --device 0")


if __name__ == "__main__":
    main()