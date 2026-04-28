"""
inference_video_fixed.py
=========================
Inference dari satu file video (.avi/.mp4) dengan pipeline BENAR.

Pipeline:
    Video → YOLO11n-pose → skeleton → Feeder preprocessing → Model → hasil

Berbeda dari inference_video.py yang lama:
    - Setelah ekstrak skeleton, masuk lewat Feeder (bukan manual preprocessing)
    - Hasilnya konsisten dengan saat training

Cara pakai (dari folder scripts/):
    python inference_video_fixed.py --input path/ke/video.avi
    python inference_video_fixed.py --input path/ke/video.avi --threshold 0.4
"""

import argparse
import importlib
import os
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Default ────────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS   = "../weights/8/runs-43-6149.pt"
DEFAULT_CONFIG    = "../config/fall-detection/balanced.yaml"
DEFAULT_YOLO      = "yolo11n-pose.pt"
DEFAULT_VIDEO     = "../dataset/ntu_videos/S013C003P027R002A008_rgb.avi"
DEFAULT_THRESHOLD = 0.5
DEFAULT_MAX_FRAMES = 300

NUM_JOINTS  = 17
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


# ── Ekstraksi skeleton dari video ──────────────────────────────────────────────

def extract_skeleton(video_path, yolo_path, max_frames):
    """
    Ekstrak skeleton dari video menggunakan YOLO11n-pose.
    Return: numpy array (T, 17, 3) — [x, y, confidence] per joint
    """
    try:
        from ultralytics import YOLO
        import cv2
    except ImportError:
        raise ImportError("pip install ultralytics opencv-python")

    print(f"  Memuat YOLO dari: {yolo_path}")
    yolo = YOLO(yolo_path)

    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Video: {total} frame, {fps:.1f} fps, {total/max(fps,1):.1f} detik")

    if total > max_frames:
        take = set(map(int, np.linspace(0, total - 1, max_frames)))
    else:
        take = set(range(total))

    seq, bad, fi = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi in take:
            kf = np.zeros((NUM_JOINTS, 3), np.float32)
            r  = yolo(frame, verbose=False)
            if r and r[0].keypoints is not None and len(r[0].keypoints) > 0:
                kps  = r[0].keypoints
                conf = kps.conf.cpu().numpy()
                best = int(conf.mean(axis=1).argmax())
                kf[:, :2] = kps.xy.cpu().numpy()[best]
                kf[:, 2]  = conf[best]
            else:
                bad += 1
            seq.append(kf)
        fi += 1
    cap.release()

    if not seq:
        raise ValueError("Tidak ada frame yang berhasil diproses!")

    bad_ratio = bad / len(seq)
    print(f"  Deteksi berhasil: {len(seq)-bad}/{len(seq)} frame ({bad_ratio*100:.1f}% gagal)")
    if bad_ratio > 0.5:
        print("  [WARN] >50% frame gagal — hasil mungkin tidak akurat")

    return np.stack(seq)   # (T, 17, 3)


def normalize_skeleton(sk):
    """
    Normalisasi ke koordinat relatif tubuh.
    - Translasi: hip center → origin
    - Scale: jarak bahu = 1
    """
    sk = sk.copy()
    xy = sk[:, :, :2]
    hc = (xy[:, 11] + xy[:, 12]) / 2.0
    xy -= hc[:, np.newaxis, :]
    d  = np.linalg.norm(xy[:, 5] - xy[:, 6], axis=1)
    sc = d[d > 1e-5].mean() if (d > 1e-5).any() else 1.0
    if sc > 1e-5:
        xy /= sc
    sk[:, :, :2] = xy
    return sk


# ── Inference via Feeder ───────────────────────────────────────────────────────

def inference_via_feeder(cfg, skeleton_np, weights_path, device, threshold):
    """
    Bungkus skeleton ke format dataset sementara,
    load lewat Feeder yang identik dengan training,
    jalankan model, return prediksi.
    """
    # Format untuk dataset: (1, C, T, V, M)
    sk  = skeleton_np.astype(np.float32)   # (T, 17, 3)
    arr = sk.transpose(2, 0, 1)[np.newaxis, :, :, :, np.newaxis]  # (1, 3, T, 17, 1)

    # Buat file sementara
    tmp = Path(tempfile.mkdtemp())
    f_data = str(tmp / "d.npy")
    f_lbl  = str(tmp / "l.pkl")

    np.save(f_data, arr)
    with open(f_lbl, "wb") as f:
        pickle.dump((["video"], [0]), f)

    # Load lewat Feeder
    Feeder      = import_class(cfg["feeder"])
    feeder_args = dict(cfg["test_feeder_args"])
    feeder_args.update({
        "data_path":    f_data,
        "label_path":   f_lbl,
        "split":        "val",
        "use_mmap":     False,
        "random_move":  False,
        "random_shift": False,
        "random_flip":  False,
        "random_speed": False,
    })

    try:
        dataset = Feeder(**feeder_args)
        x, _   = dataset[0]                   # (C, T, V, M) tensor
    finally:
        os.remove(f_data)
        os.remove(f_lbl)
        tmp.rmdir()

    # Load model dan inference
    model = load_model(cfg, weights_path, device)
    x     = x.unsqueeze(0).float().to(device)  # (1, C, T, V, M)

    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0]

    p_not_fall = probs[0].item()
    p_fall     = probs[1].item()
    pred       = 1 if p_fall >= threshold else 0

    return pred, p_fall, p_not_fall


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Inference fall detection dari satu video")
    ap.add_argument("--input",      default=DEFAULT_VIDEO)
    ap.add_argument("--weights",    default=DEFAULT_WEIGHTS)
    ap.add_argument("--config",     default=DEFAULT_CONFIG)
    ap.add_argument("--yolo_model", default=DEFAULT_YOLO)
    ap.add_argument("--threshold",  type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--max_frames", type=int,   default=DEFAULT_MAX_FRAMES)
    ap.add_argument("--device",     default="cuda:0")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA tidak tersedia, pakai CPU")
        args.device = "cpu"

    video_path = Path(args.input)
    if not video_path.exists():
        print(f"ERROR: File tidak ditemukan: {video_path}")
        sys.exit(1)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print("\n" + "=" * 55)
    print("  Fall Detection — Inference Video")
    print("=" * 55)
    print(f"  Video    : {video_path.name}")
    print(f"  Threshold: {args.threshold}")

    # 1. Ekstrak skeleton
    print("\n[1/2] Ekstrak skeleton ...")
    sk = extract_skeleton(video_path, args.yolo_model, args.max_frames)
    print(f"  Skeleton: {sk.shape}  range=[{sk[:,:,:2].min():.2f}, {sk[:,:,:2].max():.2f}]")

    # 2. Normalisasi
    sk = normalize_skeleton(sk)
    print(f"  Setelah normalisasi: range=[{sk[:,:,:2].min():.2f}, {sk[:,:,:2].max():.2f}]")

    # 3. Inference via Feeder
    print("\n[2/2] Inference ...")
    pred, p_fall, p_not_fall = inference_via_feeder(
        cfg, sk, args.weights, args.device, args.threshold)

    # Tampilkan hasil
    bar_f  = "█" * int(p_fall * 40)
    bar_nf = "█" * int(p_not_fall * 40)

    print("\n" + "=" * 55)
    print("  HASIL")
    print("=" * 55)
    print(f"  P(not_fall): {p_not_fall*100:5.1f}%  {bar_nf}")
    print(f"  P(fall)    : {p_fall*100:5.1f}%  {bar_f}")
    print()
    print(f"  >>> {LABEL_NAMES[pred]}")
    print("=" * 55)

    if 0.35 <= p_fall <= 0.65:
        print(f"\n  [INFO] Score mendekati threshold ({args.threshold}).")
        print(f"         Coba --threshold 0.4 untuk recall lebih tinggi.")


if __name__ == "__main__":
    main()