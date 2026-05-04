"""
prepare_dataset_v2.py
=====================
Versi baru prepare_dataset berbasis CSV catalog.

Perbedaan utama dari prepare_dataset.py:
- Split (train/val) dicatat eksplisit di catalog.csv → bisa diinspeksi, diedit, di-commit
- CSV berisi metadata lengkap: filename, split, actioncode, kelas_ntu, label, frame, c, v, t, m
- Dua tahap terpisah: "catalog" (buat CSV) dan "build" (bangun tensor dari CSV)
- Output tetap kompatibel dengan feeder_yolo.py (.npy + .pkl)

Cara pakai:
    # Tahap 1: buat catalog.csv (sekali jalan, bisa diedit manual sesudahnya)
    python prepare_dataset_v2.py --step catalog \\
        --skeleton_dir ../dataset/ntu_skeleton \\
        --catalog      ../dataset/catalog.csv

    # Tahap 2: bangun tensor dari catalog.csv
    python prepare_dataset_v2.py --step build \\
        --skeleton_dir ../dataset/ntu_skeleton \\
        --catalog      ../dataset/catalog.csv \\
        --out_dir      ../dataset/ntu_data

    # Atau sekaligus (default):
    python prepare_dataset_v2.py --skeleton_dir ../dataset/ntu_skeleton

Kenapa bukan satu .npz seperti repo asli BlockGCN?
    Repo asli pakai NTU120 yang sudah punya split resmi (cross-subject: pembagian
    berdasarkan ID subjek). Script data_gen/ mereka mengumpulkan semua sample ke
    satu array besar lalu simpan ke .npz dengan key x_train/y_train/x_test/y_test.
    Di sini data diekstrak manual via YOLO11-pose tanpa split bawaan, jadi split
    dihitung dari awal. File .npy terpisah + mmap_mode="r" lebih hemat RAM untuk
    dataset besar, dan .pkl menyimpan sample_names yang dibutuhkan test script.
"""

import argparse
import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    sys.exit("pip install scikit-learn")

NUM_JOINTS = 17
CHANNELS   = 3
NUM_PERSON = 1

# Action code NTU → (nama kelas NTU, label biner)
ACTION_MAP = {
    "A008": ("sitting_down",  "not_fall"),
    "A009": ("standing_up",   "not_fall"),
    "A027": ("jump_up",       "not_fall"),
    "A042": ("staggering",    "not_fall"),
    "A043": ("falling_down",  "fall"),
}

LABEL_INT = {"not_fall": 0, "fall": 1}

BONE_PAIRS = [
    (1, 0), (2, 0), (3, 1), (4, 2),
    (5, 0), (6, 0), (7, 5), (8, 6), (9, 7), (10, 8),
    (11, 5), (12, 6), (13, 11), (14, 12), (15, 13), (16, 14),
]


# ── Preprocessing ──────────────────────────────────────────────────────────────

def parse_action_code(stem: str) -> str:
    m = re.search(r'(A\d{3})', stem, re.IGNORECASE)
    return m.group(1).upper() if m else "UNKNOWN"


def pad_or_crop(seq: np.ndarray, target: int) -> np.ndarray:
    """(T,V,C) → (target,V,C). Pad nol jika kurang, crop tengah jika lebih."""
    T = seq.shape[0]
    if T == target:
        return seq
    if T < target:
        pad = np.zeros((target - T, NUM_JOINTS, CHANNELS), np.float32)
        return np.concatenate([seq, pad], axis=0)
    start = (T - target) // 2
    return seq[start: start + target]


def to_tensor(seq: np.ndarray, target: int) -> np.ndarray:
    """(T,V,C) → (C,T,V,1)  — format yang diharapkan feeder_yolo.py"""
    s = pad_or_crop(seq, target)
    return s.transpose(2, 0, 1)[:, :, :, np.newaxis]


def build_bone(seq: np.ndarray) -> np.ndarray:
    """(T,17,3) → bone modality (displacement parent→child)."""
    b = np.zeros_like(seq)
    for child, parent in BONE_PAIRS:
        b[:, child, :2] = seq[:, child, :2] - seq[:, parent, :2]
        b[:, child, 2]  = np.minimum(seq[:, child, 2], seq[:, parent, 2])
    return b


def build_motion(seq: np.ndarray) -> np.ndarray:
    """(T,17,3) → motion modality (frame-diff, last frame = 0)."""
    m = np.zeros_like(seq)
    m[:-1] = seq[1:] - seq[:-1]
    return m


# ── Step 1: Catalog ────────────────────────────────────────────────────────────

def cmd_catalog(args):
    skel    = Path(args.skeleton_dir)
    out_csv = Path(args.catalog)

    rows = []
    for label_dir in ["fall", "not_fall"]:
        folder = skel / label_dir
        if not folder.exists():
            print(f"  [WARN] Folder tidak ada: {folder}")
            continue

        files = sorted(folder.glob("*.npy"))
        print(f"  {label_dir:10s}: {len(files)} file")

        for fp in files:
            arr = np.load(str(fp))
            if arr.ndim != 3 or arr.shape[1] != NUM_JOINTS:
                print(f"    [SKIP] shape salah {arr.shape}: {fp.name}")
                continue

            T  = arr.shape[0]
            ac = parse_action_code(fp.stem)
            kelas_ntu, label_biner = ACTION_MAP.get(ac, ("unknown", label_dir))

            rows.append({
                "filename":   fp.stem,
                "split":      "",           # diisi setelah stratified split
                "actioncode": ac,
                "kelas_ntu":  kelas_ntu,
                "label":      label_biner,
                "frame":      T,            # jumlah frame aktual sebelum padding
                "c":          CHANNELS,
                "v":          NUM_JOINTS,
                "t":          args.max_frames,  # target setelah pad/crop
                "m":          NUM_PERSON,
            })

    if not rows:
        sys.exit(f"Tidak ada sample! Pastikan {skel}/fall/ dan {skel}/not_fall/ ada.")

    df = pd.DataFrame(rows)

    # Stratified split berdasarkan label biner
    labels_all = df["label"].tolist()
    tr_idx, va_idx = train_test_split(
        df.index.tolist(),
        test_size=args.val_split,
        stratify=labels_all,
        random_state=args.seed,
    )
    df.loc[tr_idx, "split"] = "train"
    df.loc[va_idx, "split"] = "val"

    # Pastikan urutan kolom sesuai permintaan
    col_order = ["filename", "split", "actioncode", "kelas_ntu", "label",
                 "frame", "c", "v", "t", "m"]
    df = df[col_order].sort_values(["split", "label", "filename"]).reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out_csv), index=False)

    print(f"\nCatalog disimpan: {out_csv}")
    print(f"Total: {len(df)} sample\n")

    summary = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    print(summary.to_string())

    print(f"\nActioncode distribution:")
    print(df.groupby(["actioncode", "kelas_ntu", "label"]).size()
            .reset_index(name="count").to_string(index=False))

    print(f"""
Tip: Bisa edit {out_csv} secara manual untuk mengubah split assignment
     (misalnya memindahkan subjek tertentu ke val), lalu jalankan --step build.
""")


# ── Step 2: Build ──────────────────────────────────────────────────────────────

def _save_split(out_dir: Path, split: str, data: np.ndarray,
                labels: list, names: list):
    np.save(str(out_dir / f"{split}_data.npy"), data)
    with open(out_dir / f"{split}_label.pkl", "wb") as f:
        pickle.dump((names, labels), f)
    c = Counter(labels)
    print(f"    {split}_data.npy  shape={data.shape}  "
          f"not_fall={c[0]}  fall={c[1]}")


def _build_version(version_name: str, out_dir: Path, skel: Path,
                   df_train: pd.DataFrame, df_val: pd.DataFrame,
                   max_frames: int, transform=None):
    vout = out_dir / version_name
    vout.mkdir(parents=True, exist_ok=True)
    print(f"\n  [{version_name}]")

    for split_name, df_split in [("train", df_train), ("val", df_val)]:
        N    = len(df_split)
        data = np.zeros((N, CHANNELS, max_frames, NUM_JOINTS, NUM_PERSON), np.float32)
        labels, names = [], []

        for i, row in enumerate(df_split.itertuples(index=False)):
            label_dir = "fall" if row.label == "fall" else "not_fall"
            fp = skel / label_dir / f"{row.filename}.npy"
            if not fp.exists():
                print(f"      [WARN] File tidak ada: {fp}")
                continue
            arr = np.load(str(fp)).astype(np.float32)
            if transform:
                arr = transform(arr)
            data[i] = to_tensor(arr, max_frames)
            labels.append(LABEL_INT[row.label])
            names.append(row.filename)

        _save_split(vout, split_name, data, labels, names)

    # dataset_info.json
    tc = Counter(LABEL_INT[r.label] for r in df_train.itertuples(index=False))
    vc = Counter(LABEL_INT[r.label] for r in df_val.itertuples(index=False))
    pw = round(tc[0] / max(tc[1], 1), 4)
    info = {
        "version":    version_name,
        "num_class":  2,
        "classes":    {"0": "not_fall", "1": "fall"},
        "num_joints": NUM_JOINTS,
        "channels":   CHANNELS,
        "max_frames": max_frames,
        "num_person": NUM_PERSON,
        "train": {"total": len(df_train), "not_fall": tc[0], "fall": tc[1]},
        "val":   {"total": len(df_val),   "not_fall": vc[0], "fall": vc[1]},
        "pos_weight": pw,
        "class_weight_for_config": [1.0, round(pw, 2)],
    }
    with open(vout / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"    pos_weight={pw}  → class_weight: [1.0, {round(pw, 2)}]")


def cmd_build(args):
    skel    = Path(args.skeleton_dir)
    catalog = Path(args.catalog)
    out     = Path(args.out_dir)

    if not catalog.exists():
        sys.exit(f"Catalog tidak ditemukan: {catalog}\nJalankan --step catalog terlebih dahulu.")

    df = pd.read_csv(str(catalog))
    print(f"Membaca catalog: {catalog}  ({len(df)} baris)")

    unknown = df[df.actioncode == "UNKNOWN"]
    if len(unknown):
        print(f"  [WARN] {len(unknown)} sample dengan actioncode UNKNOWN — pastikan "
              f"nama file mengandung Axxx (contoh: S001C001P001R001A043_rgb.npy)")

    df_train = df[df.split == "train"].reset_index(drop=True)
    df_val   = df[df.split == "val"].reset_index(drop=True)
    print(f"Train: {len(df_train)}  Val: {len(df_val)}")

    # ── BALANCED: undersample not_fall di training ────────────────────────────
    if not args.skip_balanced:
        print(f"\n{'─'*50}")
        print("Versi BALANCED (undersample not_fall di train)")
        fall_tr    = df_train[df_train.label == "fall"]
        notfall_tr = df_train[df_train.label == "not_fall"].sample(
            n=len(fall_tr), random_state=args.seed)
        df_bal = pd.concat([fall_tr, notfall_tr]).sample(
            frac=1, random_state=args.seed).reset_index(drop=True)
        print(f"  Train balanced: fall={len(fall_tr)} not_fall={len(notfall_tr)}")

        for mod, fn in [("balanced/joint", None),
                        ("balanced/bone",  build_bone),
                        ("balanced/motion", build_motion)]:
            _build_version(mod, out, skel, df_bal, df_val, args.max_frames, fn)

    # ── FULL: semua data + class_weight ──────────────────────────────────────
    print(f"\n{'─'*50}")
    print("Versi FULL (semua train data + class_weight)")
    for mod, fn in [("full/joint", None),
                    ("full/bone",  build_bone),
                    ("full/motion", build_motion)]:
        _build_version(mod, out, skel, df_train, df_val, args.max_frames, fn)

    # Tampilkan class_weight untuk config
    with open(out / "full" / "joint" / "dataset_info.json") as f:
        pw = json.load(f)["pos_weight"]

    print(f"\n{'='*50}")
    print("SELESAI")
    print(f"\nUntuk config/fall_detection/full.yaml:")
    print(f"  loss_args:")
    print(f"    weight: [1.0, {pw}]")
    print(f"\nTraining:")
    print(f"  python main.py --config config/fall_detection/balanced.yaml "
          f"--work-dir work_dir/fall_balanced --device 0")
    print(f"  python main.py --config config/fall_detection/full.yaml "
          f"--work-dir work_dir/fall_full --device 0")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Prepare fall detection dataset dengan CSV catalog.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", choices=["catalog", "build", "all"], default="all",
                    help="catalog=buat CSV saja, build=bangun tensor dari CSV, all=keduanya (default)")
    ap.add_argument("--skeleton_dir", default="../dataset/ntu_skeleton",
                    help="Folder berisi subfolder fall/ dan not_fall/")
    ap.add_argument("--catalog",      default="../dataset/catalog.csv",
                    help="Path output/input catalog CSV")
    ap.add_argument("--out_dir",      default="../dataset/ntu_data",
                    help="Folder output tensor dataset")
    ap.add_argument("--max_frames",   type=int,   default=150,
                    help="Jumlah frame setelah pad/crop (kolom 't' di catalog)")
    ap.add_argument("--val_split",    type=float, default=0.2,
                    help="Proporsi val set (default 0.2 = 80/20)")
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--skip_balanced", action="store_true",
                    help="Lewati pembuatan versi balanced (hanya buat full)")
    args = ap.parse_args()

    if args.step in ("catalog", "all"):
        print(f"\n{'='*50}")
        print("Step 1: Membuat catalog CSV")
        print(f"{'='*50}")
        cmd_catalog(args)

    if args.step in ("build", "all"):
        print(f"\n{'='*50}")
        print("Step 2: Membangun tensor dataset")
        print(f"{'='*50}")
        cmd_build(args)


if __name__ == "__main__":
    main()
