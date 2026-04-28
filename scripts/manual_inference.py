"""
inference_single.py
====================
Inference satu file skeleton .npy menggunakan pipeline
yang IDENTIK dengan training (lewat Feeder).

Ini adalah cara yang BENAR untuk inference satu sampel.
Hasilnya konsisten dengan test_fall_final.py.

Cara pakai (dari folder scripts/):
    python inference_single.py --input ../dataset/ntu_skeleton/not_fall/S001C001P001R001A008_rgb.npy
    python inference_single.py --input ../dataset/ntu_skeleton/fall/S001C001P001R001A043_rgb.npy

Atau langsung jalankan tanpa argumen (pakai DEFAULT_INPUT di bawah):
    python inference_single.py
"""

import argparse
import importlib
import pickle
import sys
import tempfile
import os
from pathlib import Path

import numpy as np
import torch
import yaml

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Default — sesuaikan ────────────────────────────────────────────────────────
DEFAULT_INPUT   = "../dataset/ntu_skeleton/not_fall/S013C003P027R002A008_rgb.npy"
DEFAULT_WEIGHTS = "../weights/8/runs-43-6149.pt"
DEFAULT_CONFIG  = "../config/fall-detection/balanced.yaml"

LABEL_NAMES = {0: "TIDAK JATUH", 1: "** JATUH **"}
LABEL_SHORT = {0: "not_fall",    1: "fall"}


def import_class(name):
    parts = name.split('.')
    mod   = importlib.import_module('.'.join(parts[:-1]))
    return getattr(mod, parts[-1])


def load_model(cfg, weights_path, device):
    Model   = import_class(cfg["model"])
    model   = Model(**cfg["model_args"])
    weights = torch.load(weights_path, map_location="cpu", weights_only=False)
    weights = {k.replace("module.", ""): v for k, v in weights.items()}
    model.load_state_dict(weights)
    model.to(device).eval()
    return model


def inference_with_feeder(cfg, skeleton_npy, weights_path, device, threshold=0.5):
    """
    Cara BENAR: bungkus skeleton ke format dataset sementara,
    lalu load lewat Feeder yang identik dengan training.

    Langkah:
    1. Simpan skeleton ke file .npy sementara (format dataset)
    2. Buat label .pkl sementara
    3. Load lewat Feeder (preprocessing identik dengan training)
    4. Jalankan model
    5. Hapus file sementara
    """

    # ── 1. Siapkan skeleton dalam format dataset (C, T, V, M) ─────────────────
    # skeleton_npy shape: (T, 17, 3)
    sk = skeleton_npy.astype(np.float32)
    T, V, C = sk.shape

    # Transpose ke (C, T, V) lalu tambah batch N=1 dan M=1
    # Shape: (1, 3, T, 17, 1) — sama dengan format train_data.npy
    data_tensor = sk.transpose(2, 0, 1)[np.newaxis, :, :, :, np.newaxis]

    # ── 2. Buat file sementara ─────────────────────────────────────────────────
    tmp_dir  = Path(tempfile.mkdtemp())
    tmp_data = str(tmp_dir / "tmp_data.npy")
    tmp_lbl  = str(tmp_dir / "tmp_label.pkl")

    np.save(tmp_data, data_tensor)   # shape (1, 3, T, 17, 1)

    with open(tmp_lbl, "wb") as f:
        pickle.dump((["single_sample"], [0]), f)   # label dummy = 0

    # ── 3. Load lewat Feeder ───────────────────────────────────────────────────
    Feeder      = import_class(cfg["feeder"])
    feeder_args = dict(cfg["test_feeder_args"])

    # Override path ke file sementara
    feeder_args["data_path"]   = tmp_data
    feeder_args["label_path"]  = tmp_lbl
    feeder_args["split"]       = "val"
    feeder_args["use_mmap"]    = False

    # Pastikan tidak ada augmentasi
    feeder_args["random_move"]  = False
    feeder_args["random_shift"] = False
    feeder_args["random_flip"]  = False
    feeder_args["random_speed"] = False

    try:
        dataset = Feeder(**feeder_args)
        x, _   = dataset[0]                          # (C, T, V, M)
    finally:
        # Bersihkan file sementara
        os.remove(tmp_data)
        os.remove(tmp_lbl)
        tmp_dir.rmdir()

    # ── 4. Load model dan inferensi ───────────────────────────────────────────
    model = load_model(cfg, weights_path, device)

    x = x.unsqueeze(0).float().to(device)             # (1, C, T, V, M)

    with torch.no_grad():
        logits = model(x)                             # (1, 2)
        probs  = torch.softmax(logits, dim=1)[0]     # (2,)

    prob_not_fall = probs[0].item()
    prob_fall     = probs[1].item()
    pred          = 1 if prob_fall >= threshold else 0

    return pred, prob_fall, prob_not_fall


def main():
    ap = argparse.ArgumentParser(
        description="Inference satu skeleton .npy menggunakan Feeder yang sama dengan training")
    ap.add_argument("--input",     default=DEFAULT_INPUT,
                    help="Path ke file skeleton .npy (shape T,17,3)")
    ap.add_argument("--weights",   default=DEFAULT_WEIGHTS,
                    help="Path ke file .pt model")
    ap.add_argument("--config",    default=DEFAULT_CONFIG,
                    help="Path ke config YAML")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device",    default="cuda:0")
    args = ap.parse_args()

    # Validasi device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA tidak tersedia, pakai CPU")
        args.device = "cpu"

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: File tidak ditemukan: {input_path}")
        sys.exit(1)

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("\n" + "=" * 55)
    print("  Fall Detection — Inference Single File")
    print("=" * 55)
    print(f"  Input    : {input_path.name}")
    print(f"  Weights  : {Path(args.weights).name}")
    print(f"  Threshold: {args.threshold}")

    # Load skeleton
    sk = np.load(str(input_path))
    print(f"\n  Skeleton shape: {sk.shape}")

    # Validasi shape
    if sk.ndim == 5:
        # Sudah dalam format dataset (1, C, T, V, M) — konversi balik ke (T, V, C)
        sk = sk[0, :, :, :, 0].transpose(1, 2, 0)
        print(f"  Converted to: {sk.shape}")
    elif sk.ndim != 3 or sk.shape[1] != 17:
        print(f"  ERROR: Shape tidak valid. Ekspektasi (T, 17, 3), dapat {sk.shape}")
        sys.exit(1)

    print(f"  Range koordinat: min={sk[:,:,:2].min():.3f}, max={sk[:,:,:2].max():.3f}")

    # Inference menggunakan Feeder
    print("\n  Menjalankan inference via Feeder ...")
    pred, prob_fall, prob_not_fall = inference_with_feeder(
        cfg, sk, args.weights, args.device, args.threshold)

    # Tampilkan hasil
    bar_fall    = "█" * int(prob_fall * 40)
    bar_notfall = "█" * int(prob_not_fall * 40)

    print("\n" + "=" * 55)
    print("  HASIL PREDIKSI")
    print("=" * 55)
    print(f"  File       : {input_path.name}")
    print(f"  Threshold  : {args.threshold}")
    print()
    print(f"  P(not_fall): {prob_not_fall*100:5.1f}%  {bar_notfall}")
    print(f"  P(fall)    : {prob_fall*100:5.1f}%  {bar_fall}")
    print()
    print(f"  >>> PREDIKSI: {LABEL_NAMES[pred]}")
    print("=" * 55)

    # Cek label dari nama file (dari NTU naming convention)
    filename = input_path.stem
    if "A043" in filename:
        true_label = "fall"
    elif any(f"A{c:03d}" in filename for c in [8, 9, 27, 42]):
        true_label = "not_fall"
    else:
        true_label = "unknown"

    if true_label != "unknown":
        pred_str = LABEL_SHORT[pred]
        status   = "BENAR" if pred_str == true_label else "SALAH"
        print(f"\n  Label asli  : {true_label}")
        print(f"  Prediksi    : {pred_str}")
        print(f"  Status      : {status}")


if __name__ == "__main__":
    main()