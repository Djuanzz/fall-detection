"""
realtime_ntu_inference.py
=========================
Simulasi inferensi real-time dari file .skeleton NTU RGB+D.
Memproses file frame-per-frame dengan sliding window, persis seperti
yang akan dilakukan pada stream Kinect live.

Untuk Kinect LIVE: uncomment bagian "Kinect Live Mode" di bawah dan
install pyk4a (Azure Kinect) atau pykinect2 (Kinect v2).

Cara pakai (simulasi dari file):
    python scripts/realtime_ntu_inference.py \\
        --skeleton  /path/to/S001C001P001R001A043.skeleton \\
        --weights   work_dir/fall_ntu25_balanced/runs-90-XXXX.pt \\
        --config    config/fall-detection-ntu/balanced.yaml \\
        --fps       30 \\
        --window    150 \\
        --step      15
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.prepare_dataset_ntu25 import (
    read_skeleton_file,
    select_primary_body,
    normalize_skeleton,
    CHANNELS,
    NUM_JOINTS,
)

# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(config_path: str, weights_path: str, device: torch.device):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_args = cfg.get("model_args", {})
    model_class_path = cfg.get("model", "model.BlockGCN.Model")

    components = model_class_path.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    model = getattr(mod, "Model")(**model_args)
    model = model.to(device)

    weights = torch.load(weights_path, map_location=device)
    if isinstance(weights, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in weights:
                weights = weights[key]
                break
    weights = {k.replace("module.", ""): v for k, v in weights.items()}
    model.load_state_dict(weights, strict=False)
    model.eval()
    return model


# ── Preprocessing per-frame ───────────────────────────────────────────────────

def normalize_frame(joints: np.ndarray) -> np.ndarray:
    """
    joints: (25, 3) absolute world coords
    → (25, 3) relatif ke SpineBase (joint 0)
    """
    spine = joints[0:1, :]   # (1, 3)
    return joints - spine


def window_to_tensor(window: list, target_len: int) -> np.ndarray:
    """
    window: list of (25, 3) arrays, len = window_size
    → (1, 3, target_len, 25, 1) float32 tensor
    """
    seq = np.stack(window, axis=0)           # (T, 25, 3)
    # Pad atau crop ke target_len
    T = seq.shape[0]
    if T < target_len:
        pad = np.zeros((target_len - T, NUM_JOINTS, CHANNELS), np.float32)
        seq = np.concatenate([seq, pad], axis=0)
    elif T > target_len:
        seq = seq[-target_len:]              # ambil T frame terakhir
    t = seq.transpose(2, 0, 1)              # (3, T, 25)
    t = t[:, :, :, np.newaxis]              # (3, T, 25, 1)
    return t[np.newaxis].astype(np.float32) # (1, 3, T, 25, 1)


# ── Inferensi satu window ──────────────────────────────────────────────────────

def infer_window(model, tensor: np.ndarray, device: torch.device, threshold: float):
    x = torch.tensor(tensor, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    fall_prob = float(probs[1])
    pred = 1 if fall_prob >= threshold else 0
    return pred, fall_prob


# ── Visualisasi terminal ───────────────────────────────────────────────────────

def print_status(frame_idx: int, pred: int, fall_prob: float,
                 gt_label: int = None, alert_history: list = None):
    label_str = "!!! FALL DETECTED !!!" if pred == 1 else "    not fall       "
    bar_len = 30
    filled = int(fall_prob * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)

    gt_str = ""
    if gt_label is not None:
        gt_str = f"  GT:{('FALL' if gt_label==1 else 'NOT_FALL')}"

    status = (f"\r[Frame {frame_idx:5d}]  P(fall)={fall_prob:.4f}  "
              f"[{bar}]  {label_str}{gt_str}  ")
    print(status, end="", flush=True)


# ── Mode simulasi file ─────────────────────────────────────────────────────────

def run_file_simulation(args, model, device):
    """Baca .skeleton frame-per-frame, simulasi sliding window real-time."""
    print(f"\nMemuat file skeleton: {args.skeleton}")
    _, frames = read_skeleton_file(args.skeleton)

    # Ground truth dari nama file
    act = -1
    stem = Path(args.skeleton).stem
    try:
        a_parts = [p for p in stem.split('A') if len(p) >= 3 and p[:3].isdigit()]
        act = int(a_parts[-1][:3])
    except (IndexError, ValueError):
        pass
    gt_label = 1 if act == 43 else (0 if act in {8, 9, 27, 42} else None)
    gt_str = f"A{act:03d} → {'FALL' if gt_label==1 else 'NOT_FALL'}" if gt_label is not None else "?"
    print(f"Ground truth: {gt_str}")

    total_frames = len(frames)
    print(f"Total frame : {total_frames}")
    print(f"Window size : {args.window}  Step: {args.step}")
    print(f"Threshold   : {args.threshold}")
    print(f"Simulated FPS: {args.fps}")
    print("\nMulai inferensi... (Ctrl+C untuk berhenti)\n")

    frame_delay = 1.0 / args.fps if args.fps > 0 else 0
    window = deque(maxlen=args.window)
    predictions = []
    last_pred = 0
    last_fall_prob = 0.0

    try:
        for frame_idx, frame_bodies in enumerate(frames):
            # Ambil skeleton dari frame ini (primary body)
            if frame_bodies:
                joints = frame_bodies[0]['joints']   # (25, 3)
            else:
                joints = np.zeros((NUM_JOINTS, CHANNELS), np.float32)

            joints_norm = normalize_frame(joints)
            window.append(joints_norm)

            # Jalankan inferensi setiap args.step frame jika window penuh
            if len(window) == args.window and (frame_idx % args.step == 0):
                tensor = window_to_tensor(list(window), args.window)
                pred, fall_prob = infer_window(model, tensor, device, args.threshold)
                last_pred = pred
                last_fall_prob = fall_prob
                predictions.append((frame_idx, pred, fall_prob))

            print_status(frame_idx, last_pred, last_fall_prob, gt_label)

            if frame_delay > 0:
                time.sleep(frame_delay)

    except KeyboardInterrupt:
        print("\n\nDihentikan oleh user.")

    # Ringkasan
    print("\n")
    if predictions:
        fall_frames = sum(1 for _, p, _ in predictions if p == 1)
        total_infer = len(predictions)
        fall_ratio  = fall_frames / total_infer if total_infer > 0 else 0
        avg_prob    = sum(fp for _, _, fp in predictions) / total_infer
        max_prob    = max(fp for _, _, fp in predictions)

        print(f"{'─'*50}")
        print(f"Ringkasan inferensi:")
        print(f"  Total window inferred : {total_infer}")
        print(f"  Window prediksi FALL  : {fall_frames} ({fall_ratio*100:.1f}%)")
        print(f"  Rata-rata P(fall)     : {avg_prob:.4f}")
        print(f"  Maks P(fall)          : {max_prob:.4f}")
        if gt_label is not None:
            verdict = "FALL" if fall_ratio > 0.3 else "NOT_FALL"
            correct = (verdict == "FALL") == (gt_label == 1)
            print(f"  Verdict (>30% FALL)  : {verdict}  ({'[BENAR]' if correct else '[SALAH]'})")


# ── Placeholder Kinect Live ────────────────────────────────────────────────────

def run_kinect_live(args, model, device):
    """
    ─────────────────────────────────────────────────
    KINECT LIVE MODE (memerlukan SDK tambahan)
    ─────────────────────────────────────────────────
    Untuk Azure Kinect (Kinect v4):
        pip install pyk4a
        Dokumentasi: https://github.com/etiennedub/pyk4a

    Untuk Kinect v2 (Windows):
        pip install pykinect2
        Dokumentasi: https://github.com/Kinect/PyKinect2

    Contoh integrasi Azure Kinect:

        import pyk4a
        from pyk4a import PyK4A, Config, ColorResolution, DepthMode

        k4a = PyK4A(Config(
            color_resolution=ColorResolution.RES_720P,
            depth_mode=DepthMode.NFOV_UNBINNED,
        ))
        k4a.start()

        window = deque(maxlen=args.window)
        frame_idx = 0

        while True:
            capture = k4a.get_capture()
            if capture.body_index_map is None:
                continue

            # Ekstrak 25 joint dari skeleton (Azure Kinect = 32 joints)
            # Perlu mapping dari AK 32-joint ke NTU 25-joint

            joints = extract_ntu_joints_from_azure_kinect(capture)
            joints_norm = normalize_frame(joints)
            window.append(joints_norm)

            if len(window) == args.window and frame_idx % args.step == 0:
                tensor = window_to_tensor(list(window), args.window)
                pred, fall_prob = infer_window(model, tensor, device, args.threshold)
                print_status(frame_idx, pred, fall_prob)

            frame_idx += 1

        k4a.stop()
    ─────────────────────────────────────────────────
    """
    print("[INFO] Kinect Live Mode belum diimplementasikan.")
    print("       Lihat docstring fungsi run_kinect_live() untuk panduan integrasi.")
    print("       Gunakan --mode file untuk simulasi dari file .skeleton.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Real-time fall detection dari file .skeleton NTU RGB+D")
    ap.add_argument("--skeleton",  default=None,
                    help="Path file .skeleton (untuk mode simulasi)")
    ap.add_argument("--weights",   required=True,
                    help="Path model weights (.pt)")
    ap.add_argument("--config",    required=True,
                    help="Path config YAML")
    ap.add_argument("--mode",      default="file",
                    choices=["file", "kinect"],
                    help="Mode: 'file' (simulasi) atau 'kinect' (live)")
    ap.add_argument("--window",    type=int, default=150,
                    help="Ukuran sliding window (frame, default 150)")
    ap.add_argument("--step",      type=int, default=15,
                    help="Inferensi setiap N frame (default 15 = ~0.5s pada 30fps)")
    ap.add_argument("--fps",       type=float, default=0,
                    help="Kecepatan simulasi FPS (0 = secepat mungkin)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Threshold P(fall) untuk alert (default 0.5)")
    ap.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Device  : {device}")
    print(f"Weights : {args.weights}")

    print("Loading model...")
    model = load_model(args.config, args.weights, device)
    print("Model siap.\n")

    if args.mode == "kinect":
        run_kinect_live(args, model, device)
    elif args.mode == "file":
        if not args.skeleton:
            ap.error("--skeleton diperlukan untuk mode 'file'")
        if not Path(args.skeleton).exists():
            sys.exit(f"[ERROR] File tidak ditemukan: {args.skeleton}")
        run_file_simulation(args, model, device)
    else:
        ap.error(f"Mode tidak dikenal: {args.mode}")


if __name__ == "__main__":
    main()
