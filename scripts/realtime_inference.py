"""
realtime_fall_detection.py
===========================
Sistem near real-time fall detection menggunakan webcam laptop.

Pipeline:
    Webcam → YOLO11n-pose (per frame) → buffer skeleton →
    setiap STEP_SIZE frame → BlockGCN → tampilkan prediksi

Cara pakai (dari folder scripts/):
    python realtime_fall_detection.py
    python realtime_fall_detection.py --camera 1
    python realtime_fall_detection.py --threshold 0.4 --step 10

Kontrol keyboard:
    q = keluar
    s = simpan screenshot
    r = reset buffer (mulai dari nol lagi)
"""

import argparse
import collections
import importlib
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Default konfigurasi ────────────────────────────────────────────────────────
DEFAULT_WEIGHTS   = "../weights/8/runs-43-6149.pt"
DEFAULT_CONFIG    = "../config/fall-detection/balanced.yaml"
DEFAULT_YOLO      = "yolo11n-pose.pt"
DEFAULT_CAMERA    = 0        # 0 = webcam utama laptop
DEFAULT_THRESHOLD = 0.5
DEFAULT_WINDOW    = 30      # frame yang dilihat model sekaligus
DEFAULT_STEP      = 15       # prediksi update setiap N frame baru
DEFAULT_MAX_W     = 640      # lebar display
DEFAULT_MAX_H     = 480      # tinggi display

NUM_JOINTS = 17

# Koneksi tulang untuk visualisasi (pasangan joint yang dihubungkan)
SKELETON_EDGES = [
    (0,1),(0,2),(1,3),(2,4),           # kepala
    (0,5),(0,6),(5,6),                 # bahu
    (5,7),(7,9),(6,8),(8,10),          # lengan
    (5,11),(6,12),(11,12),             # torso
    (11,13),(13,15),(12,14),(14,16),   # kaki
]

# Warna
COLOR_GREEN  = (0, 220, 0)
COLOR_RED    = (0, 0, 220)
COLOR_YELLOW = (0, 200, 200)
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0, 0, 0)
COLOR_GRAY   = (128, 128, 128)
COLOR_BLUE   = (220, 100, 0)


def import_class(name):
    parts = name.split('.')
    mod   = importlib.import_module('.'.join(parts[:-1]))
    return getattr(mod, parts[-1])


# ── Load model ─────────────────────────────────────────────────────────────────

def load_model(cfg, weights_path, device):
    Model   = import_class(cfg["model"])
    model   = Model(**cfg["model_args"])
    weights = torch.load(weights_path, map_location="cpu", weights_only=False)
    weights = {k.replace("module.", ""): v for k, v in weights.items()}
    model.load_state_dict(weights)
    model.to(device).eval()
    return model


# ── Normalisasi skeleton ───────────────────────────────────────────────────────

def normalize_skeleton_sequence(buffer_np):
    """
    Normalisasi sequence skeleton:
    - Translasi: hip center setiap frame → origin
    - Scale: rata-rata jarak bahu dari seluruh sequence

    Input:  (T, 17, 3)
    Output: (T, 17, 3)
    """
    sk = buffer_np.copy()
    xy = sk[:, :, :2]

    # Translasi per frame
    hc = (xy[:, 11] + xy[:, 12]) / 2.0    # (T, 2)
    xy -= hc[:, np.newaxis, :]

    # Scale dari seluruh sequence
    d  = np.linalg.norm(xy[:, 5] - xy[:, 6], axis=1)   # (T,)
    sc = d[d > 1e-5].mean() if (d > 1e-5).any() else 1.0
    if sc > 1e-5:
        xy /= sc

    sk[:, :, :2] = xy
    return sk


# ── Inference via Feeder ───────────────────────────────────────────────────────

def run_inference(model, cfg, skeleton_window, device, threshold):
    """
    Jalankan inference pada satu window skeleton (T, 17, 3).
    Lewat Feeder untuk preprocessing yang konsisten.

    Return: (pred, prob_fall, prob_not_fall)
    """
    # Format dataset: (1, C, T, V, M)
    sk  = skeleton_window.astype(np.float32)
    arr = sk.transpose(2, 0, 1)[np.newaxis, :, :, :, np.newaxis]

    # File sementara
    tmp    = Path(tempfile.mkdtemp())
    f_data = str(tmp / "d.npy")
    f_lbl  = str(tmp / "l.pkl")
    np.save(f_data, arr)
    with open(f_lbl, "wb") as f:
        pickle.dump((["rt"], [0]), f)

    try:
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
        dataset = Feeder(**feeder_args)
        x, _    = dataset[0]
    finally:
        os.remove(f_data)
        os.remove(f_lbl)
        tmp.rmdir()

    x = x.unsqueeze(0).float().to(device)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0]

    p_nf = probs[0].item()
    p_f  = probs[1].item()
    pred = 1 if p_f >= threshold else 0

    return pred, p_f, p_nf


# ── Visualisasi ────────────────────────────────────────────────────────────────

def draw_skeleton_on_frame(frame, keypoints_xy, keypoints_conf, conf_thresh=0.3):
    """
    Gambar skeleton (titik dan garis) di atas frame webcam.

    keypoints_xy:   (17, 2) — koordinat pixel
    keypoints_conf: (17,)   — confidence tiap joint
    """
    h, w = frame.shape[:2]

    # Gambar tulang (garis)
    for i, j in SKELETON_EDGES:
        if keypoints_conf[i] > conf_thresh and keypoints_conf[j] > conf_thresh:
            x1, y1 = int(keypoints_xy[i][0]), int(keypoints_xy[i][1])
            x2, y2 = int(keypoints_xy[j][0]), int(keypoints_xy[j][1])
            # Validasi koordinat dalam frame
            if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                cv2.line(frame, (x1, y1), (x2, y2), COLOR_BLUE, 2)

    # Gambar titik joint
    for k in range(NUM_JOINTS):
        if keypoints_conf[k] > conf_thresh:
            x, y = int(keypoints_xy[k][0]), int(keypoints_xy[k][1])
            if 0 <= x < w and 0 <= y < h:
                # Warna berdasarkan confidence
                if keypoints_conf[k] > 0.7:
                    color = COLOR_GREEN
                elif keypoints_conf[k] > 0.5:
                    color = COLOR_YELLOW
                else:
                    color = COLOR_RED
                cv2.circle(frame, (x, y), 5, color, -1)
                cv2.circle(frame, (x, y), 5, COLOR_WHITE, 1)

    return frame


def draw_status_panel(frame, pred, p_fall, p_not_fall,
                      buffer_len, window_size, fps, threshold,
                      frame_count, step_size):
    """
    Gambar panel status di bagian bawah frame.
    """
    h, w = frame.shape[:2]
    panel_h = 120

    # Background panel hitam semi-transparan
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), COLOR_BLACK, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    is_warming = buffer_len < window_size

    if is_warming:
        # Status warming up
        progress = int((buffer_len / window_size) * (w - 20))
        cv2.rectangle(frame, (10, h - panel_h + 10),
                      (10 + progress, h - panel_h + 30), COLOR_GRAY, -1)
        cv2.rectangle(frame, (10, h - panel_h + 10),
                      (w - 10, h - panel_h + 30), COLOR_WHITE, 1)

        text = f"WARMING UP... {buffer_len}/{window_size} frame"
        cv2.putText(frame, text, (15, h - panel_h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1)

        cv2.putText(frame, "Menunggu buffer penuh sebelum prediksi...",
                    (10, h - panel_h + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GRAY, 1)
    else:
        # Status prediksi
        if pred == 1:
            status_text  = "** JATUH TERDETEKSI **"
            status_color = COLOR_RED
        else:
            status_text  = "TIDAK JATUH"
            status_color = COLOR_GREEN

        # Teks status besar
        cv2.putText(frame, status_text, (10, h - panel_h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

        # Bar probabilitas fall
        bar_x, bar_y   = 10, h - panel_h + 50
        bar_w, bar_h_b = w - 20, 18
        fill_w = int(p_fall * bar_w)

        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h_b), COLOR_GRAY, -1)
        bar_color = COLOR_RED if p_fall >= threshold else COLOR_GREEN
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill_w, bar_y + bar_h_b), bar_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h_b), COLOR_WHITE, 1)

        # Garis threshold
        thr_x = bar_x + int(threshold * bar_w)
        cv2.line(frame, (thr_x, bar_y - 3), (thr_x, bar_y + bar_h_b + 3),
                 COLOR_YELLOW, 2)

        cv2.putText(frame, f"P(fall): {p_fall*100:.1f}%",
                    (bar_x, bar_y + bar_h_b + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)

    # Info kanan atas
    info_lines = [
        f"FPS: {fps:.1f}",
        f"Buf: {min(buffer_len, window_size)}/{window_size}",
        f"Thr: {threshold:.2f}",
        f"Step: {step_size}",
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(frame, line, (w - 130, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

    # Instruksi keyboard
    cv2.putText(frame, "q=keluar  s=screenshot  r=reset",
                (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GRAY, 1)

    return frame


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Near real-time fall detection via webcam")
    ap.add_argument("--weights",   default=DEFAULT_WEIGHTS)
    ap.add_argument("--config",    default=DEFAULT_CONFIG)
    ap.add_argument("--yolo",      default=DEFAULT_YOLO)
    ap.add_argument("--camera",    type=int,   default=DEFAULT_CAMERA,
                    help="Index kamera (0=webcam utama)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="Threshold probabilitas fall (0.3-0.7)")
    ap.add_argument("--window",    type=int,   default=DEFAULT_WINDOW,
                    help="Jumlah frame yang dilihat model (default 150)")
    ap.add_argument("--step",      type=int,   default=DEFAULT_STEP,
                    help="Update prediksi setiap N frame (default 15)")
    ap.add_argument("--device",    default="cuda:0")
    ap.add_argument("--width",     type=int,   default=DEFAULT_MAX_W)
    ap.add_argument("--height",    type=int,   default=DEFAULT_MAX_H)
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA tidak tersedia, pakai CPU (lebih lambat)")
        args.device = "cpu"

    # ── Load YOLO ──────────────────────────────────────────────────────────────
    print(f"\nMemuat YOLO dari: {args.yolo}")
    try:
        from ultralytics import YOLO
        yolo = YOLO(args.yolo)
    except ImportError:
        print("ERROR: pip install ultralytics")
        sys.exit(1)

    # ── Load config dan model ──────────────────────────────────────────────────
    print(f"Memuat config dari: {args.config}")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"Memuat model dari: {args.weights}")
    model = load_model(cfg, args.weights, args.device)
    print(f"Model siap di: {args.device}")

    # ── Buka webcam ────────────────────────────────────────────────────────────
    print(f"\nMembuka kamera {args.camera} ...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Kamera {args.camera} tidak bisa dibuka!")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolusi kamera: {actual_w}x{actual_h}")

    print(f"\nKonfigurasi:")
    print(f"  Window size : {args.window} frame")
    print(f"  Step size   : {args.step} frame")
    print(f"  Threshold   : {args.threshold}")
    print(f"  Update rate : setiap {args.step/30:.1f} detik (asumsi 30fps)")
    print(f"\nTekan 'q' untuk keluar, 's' untuk screenshot, 'r' untuk reset buffer")
    print("Mulai...")

    # ── State variabel ─────────────────────────────────────────────────────────
    # Buffer skeleton — deque otomatis buang yang lama kalau penuh
    skeleton_buffer = collections.deque(maxlen=args.window)

    # State prediksi terakhir
    last_pred      = 0
    last_p_fall    = 0.0
    last_p_not_fall = 1.0

    frame_count    = 0
    screenshot_num = 0

    # FPS tracking
    fps_times = collections.deque(maxlen=30)
    t_start   = time.time()

    # ── Loop utama ─────────────────────────────────────────────────────────────
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Gagal baca frame dari kamera!")
            break

        frame_count += 1
        t_now = time.time()
        fps_times.append(t_now)

        # Hitung FPS
        if len(fps_times) >= 2:
            fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0])
        else:
            fps = 0.0

        # ── 1. Deteksi skeleton dengan YOLO ────────────────────────────────────
        kf_xy   = np.zeros((NUM_JOINTS, 2), np.float32)
        kf_conf = np.zeros(NUM_JOINTS,      np.float32)

        results = yolo(frame, verbose=False)
        person_detected = False

        if results and results[0].keypoints is not None:
            kps = results[0].keypoints
            if len(kps) > 0:
                conf_all = kps.conf.cpu().numpy()        # (P, 17)
                best     = int(conf_all.mean(axis=1).argmax())
                kf_xy    = kps.xy.cpu().numpy()[best]    # (17, 2)
                kf_conf  = conf_all[best]                # (17,)
                person_detected = True

        # Buat frame skeleton (17, 3) = [x, y, confidence]
        kf = np.zeros((NUM_JOINTS, 3), np.float32)
        kf[:, :2] = kf_xy
        kf[:, 2]  = kf_conf

        # Masukkan ke buffer
        skeleton_buffer.append(kf)

        # ── 2. Gambar skeleton di frame ────────────────────────────────────────
        if person_detected:
            frame = draw_skeleton_on_frame(frame, kf_xy, kf_conf)

        # ── 3. Inference setiap STEP_SIZE frame ────────────────────────────────
        if (len(skeleton_buffer) == args.window and
                frame_count % args.step == 0):

            # Ambil window terbaru dari buffer
            window_np = np.array(skeleton_buffer)   # (150, 17, 3)

            # Normalisasi
            window_norm = normalize_skeleton_sequence(window_np)

            try:
                pred, p_fall, p_nf = run_inference(
                    model, cfg, window_norm,
                    args.device, args.threshold)

                last_pred       = pred
                last_p_fall     = p_fall
                last_p_not_fall = p_nf

            except Exception as e:
                print(f"[WARN] Inference error: {e}")

        # ── 4. Gambar panel status ─────────────────────────────────────────────
        frame = draw_status_panel(
            frame,
            pred       = last_pred,
            p_fall     = last_p_fall,
            p_not_fall = last_p_not_fall,
            buffer_len = len(skeleton_buffer),
            window_size= args.window,
            fps        = fps,
            threshold  = args.threshold,
            frame_count= frame_count,
            step_size  = args.step,
        )

        # ── 5. Tampilkan ───────────────────────────────────────────────────────
        cv2.imshow("Fall Detection — BlockGCN + YOLO11n-pose", frame)

        # ── 6. Handle keyboard ─────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("\nKeluar...")
            break

        elif key == ord('s'):
            # Simpan screenshot
            ss_path = f"screenshot_{screenshot_num:03d}.jpg"
            cv2.imwrite(ss_path, frame)
            screenshot_num += 1
            print(f"Screenshot disimpan: {ss_path}")

        elif key == ord('r'):
            # Reset buffer
            skeleton_buffer.clear()
            last_pred       = 0
            last_p_fall     = 0.0
            last_p_not_fall = 1.0
            frame_count     = 0
            print("Buffer di-reset.")

    # ── Cleanup ────────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("Selesai.")


if __name__ == "__main__":
    main()