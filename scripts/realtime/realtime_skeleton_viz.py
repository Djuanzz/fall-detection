"""
realtime_skeleton_viz.py
========================
Visualisasi real-time skeleton NTU RGB+D 25-joint dengan fall detection overlay.

Tampilan:
  - Dua panel: Front View (X-Y) dan Side View (Z-Y)
  - Skeleton 25-joint dengan warna tiap segmen tubuh
  - Confidence bar fall detection (gradient hijau -> merah)
  - Alert overlay merah saat FALL terdeteksi
  - Trajektori SpineBase untuk melihat pergerakan jatuh

Kontrol keyboard:
  SPACE   : pause / resume
  Q / ESC : keluar
  +/-     : percepat / perlambat playback
  S       : screenshot frame saat ini
  T       : toggle tampilkan trajektori

Cara pakai:
    python scripts/realtime_skeleton_viz.py \\
        --skeleton  /path/to/S001C001P001R001A043.skeleton \\
        --weights   work_dir/fall_ntu25_balanced/runs-82-15498.pt \\
        --config    config/fall-detection-ntu/balanced.yaml

Tanpa model (hanya visualisasi skeleton):
    python scripts/realtime_skeleton_viz.py \\
        --skeleton  /path/to/file.skeleton
"""

SKELETON_PATH = "e:\\000 tugasakhir\\03 code\\block-gcn-yolo\\data\\nturgbd_raw\\nturgb+d_skeletons"
WEIGHTS_PATH  = "work_dir/fall_ntu25_balanced/runs-82-15498.pt"
CONFIG_PATH   = "config/fall-detection-ntu/balanced.yaml"

import argparse
import importlib
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.prepare_dataset_ntu25 import (
    read_skeleton_file,
    normalize_skeleton,
    CHANNELS,
    NUM_JOINTS,
)

# ── Konstanta tampilan ─────────────────────────────────────────────────────────

PANEL_W   = 620    # lebar satu panel skeleton
PANEL_H   = 560    # tinggi panel skeleton
SIDE_W    = 260    # lebar sidebar info
WIN_W     = PANEL_W * 2 + SIDE_W + 4   # total lebar (~1504)
WIN_H     = PANEL_H + 80               # total tinggi (+ status bar)

# Warna BGR untuk tiap segmen tubuh
C_SPINE  = (80,  230,  80)   # hijau   : tulang belakang & kepala
C_LARM   = (255, 140,  60)   # biru    : lengan kiri
C_RARM   = (60,  140, 255)   # merah   : lengan kanan
C_LLEG   = (210, 200, 100)   # cyan    : kaki kiri
C_RLEG   = (100, 200, 210)   # kuning  : kaki kanan
C_FINGER = (180, 180, 180)   # abu     : tangan & jari
C_JOINT  = (255, 255, 255)   # putih   : semua titik joint
C_SPINE_J = (80, 230, 80)
C_HEAD_J  = (255, 200, 100)

BG_COLOR  = (20,  20,  30)   # latar belakang gelap
GRID_COLOR= (40,  40,  55)   # warna grid
TEXT_COLOR= (220, 220, 220)
FALL_COLOR= (50,  50,  255)  # merah (alert)
SAFE_COLOR= (50,  200,  50)  # hijau (aman)

# Tulang NTU 25-joint (0-indexed) dengan warna masing-masing
BONES = [
    # Spine & head
    ((0, 1),  C_SPINE), ((1, 20),  C_SPINE), ((20, 2),  C_SPINE),
    ((2, 3),   C_SPINE),
    # Left arm
    ((20, 4),  C_LARM),  ((4, 5),   C_LARM),  ((5, 6),   C_LARM),
    ((6, 7),   C_LARM),  ((7, 21),  C_FINGER), ((6, 22),  C_FINGER),
    # Right arm
    ((20, 8),  C_RARM),  ((8, 9),   C_RARM),  ((9, 10),  C_RARM),
    ((10, 11), C_RARM),  ((11, 23), C_FINGER), ((10, 24), C_FINGER),
    # Left leg
    ((0, 12),  C_LLEG),  ((12, 13), C_LLEG),  ((13, 14), C_LLEG),
    ((14, 15), C_LLEG),
    # Right leg
    ((0, 16),  C_RLEG),  ((16, 17), C_RLEG),  ((17, 18), C_RLEG),
    ((18, 19), C_RLEG),
]

# Label joint index penting untuk ditampilkan
KEY_JOINTS = {0: "Base", 2: "Neck", 3: "Head", 20: "Spine"}


# ── Model loader ───────────────────────────────────────────────────────────────

def load_model_from_config(config_path, weights_path, device):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cls_path = cfg.get("model", "model.BlockGCN.Model")
    parts = cls_path.split(".")
    mod = importlib.import_module(".".join(parts[:-1]))
    ModelClass = getattr(mod, parts[-1])
    model = ModelClass(**cfg["model_args"]).to(device)
    w = torch.load(weights_path, map_location=device)
    if isinstance(w, dict):
        for k in ("model_state_dict", "state_dict"):
            if k in w:
                w = w[k]
                break
    w = {k.replace("module.", ""): v for k, v in w.items()}
    model.load_state_dict(w, strict=False)
    model.eval()
    return model


# ── Preprocessing untuk inferensi ─────────────────────────────────────────────

def window_to_input(frames_raw, window_size):
    """
    frames_raw : list of np.ndarray (25, 3), koordinat absolut
    return     : (1, 3, T, 25, 1) float32 tensor siap untuk model
    """
    seq = np.stack(frames_raw, axis=0)          # (T, 25, 3)
    seq_norm = normalize_skeleton(seq)           # subtract SpineBase per frame

    T = seq_norm.shape[0]
    if T < window_size:
        pad = np.zeros((window_size - T, NUM_JOINTS, CHANNELS), np.float32)
        seq_norm = np.concatenate([seq_norm, pad], axis=0)
    else:
        seq_norm = seq_norm[-window_size:]

    t = seq_norm.transpose(2, 0, 1)             # (3, T, 25)
    t = t[:, :, :, np.newaxis]                  # (3, T, 25, 1)
    return torch.tensor(t[np.newaxis], dtype=torch.float32)  # (1, 3, T, 25, 1)


# ── Koordinat world → pixel ────────────────────────────────────────────────────

def world_to_pixel_front(x, y, panel_w, panel_h,
                          x_range=(-1.2, 1.2), y_range=(-0.3, 2.2)):
    """Front view: X horizontal, Y vertikal (atas = tinggi)."""
    px = int((x - x_range[0]) / (x_range[1] - x_range[0]) * panel_w)
    py = int((1.0 - (y - y_range[0]) / (y_range[1] - y_range[0])) * panel_h)
    return np.clip(px, 0, panel_w - 1), np.clip(py, 0, panel_h - 1)


def world_to_pixel_side(z, y, panel_w, panel_h,
                         z_range=(0.3, 4.5), y_range=(-0.3, 2.2)):
    """Side view: Z horizontal (dalam = kanan), Y vertikal."""
    px = int((z - z_range[0]) / (z_range[1] - z_range[0]) * panel_w)
    py = int((1.0 - (y - y_range[0]) / (y_range[1] - y_range[0])) * panel_h)
    return np.clip(px, 0, panel_w - 1), np.clip(py, 0, panel_h - 1)


# ── Renderer skeleton ──────────────────────────────────────────────────────────

def draw_grid(canvas, panel_x, panel_y, panel_w, panel_h, n=8):
    """Grid latar belakang untuk kedalaman."""
    for i in range(n + 1):
        x = panel_x + i * panel_w // n
        cv2.line(canvas, (x, panel_y), (x, panel_y + panel_h),
                 GRID_COLOR, 1, cv2.LINE_AA)
    for i in range(int(n * panel_h / panel_w) + 1):
        y = panel_y + i * panel_w // n
        if y <= panel_y + panel_h:
            cv2.line(canvas, (panel_x, y), (panel_x + panel_w, y),
                     GRID_COLOR, 1, cv2.LINE_AA)


def draw_skeleton(canvas, joints, offset_x, offset_y, view="front",
                  alpha=1.0, is_valid=True):
    """
    Gambar skeleton 25-joint pada canvas.
    joints   : (25, 3) world coordinates
    offset_x : pixel x start panel
    view     : 'front' (XY) atau 'side' (ZY)
    """
    if not is_valid:
        return

    pw = PANEL_W
    ph = PANEL_H

    def to_px(j_idx):
        x_w, y_w, z_w = joints[j_idx]
        if view == "front":
            return world_to_pixel_front(x_w, y_w, pw, ph)
        else:
            return world_to_pixel_side(z_w, y_w, pw, ph)

    # Gambar tulang (bones)
    for (j1, j2), color in BONES:
        if np.all(joints[j1] == 0) or np.all(joints[j2] == 0):
            continue
        p1 = (offset_x + to_px(j1)[0], offset_y + to_px(j1)[1])
        p2 = (offset_x + to_px(j2)[0], offset_y + to_px(j2)[1])
        cv2.line(canvas, p1, p2, color, 3, cv2.LINE_AA)

    # Gambar joint (titik)
    for j in range(NUM_JOINTS):
        if np.all(joints[j] == 0):
            continue
        px, py = to_px(j)
        px += offset_x
        py += offset_y
        r = 6 if j in (3, 20) else 4  # head & spine lebih besar
        c = C_HEAD_J if j == 3 else C_SPINE_J if j in (0, 1, 2, 20) else C_JOINT
        cv2.circle(canvas, (px, py), r, c, -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), r + 1, (0, 0, 0), 1, cv2.LINE_AA)


def draw_trajectory(canvas, traj, offset_x, offset_y, view="front", color=(200, 200, 50)):
    """Gambar trajektori SpineBase (joint 0)."""
    if len(traj) < 2:
        return
    pts = []
    for joints in traj:
        x_w, y_w, z_w = joints[0]  # SpineBase
        if view == "front":
            px, py = world_to_pixel_front(x_w, y_w, PANEL_W, PANEL_H)
        else:
            px, py = world_to_pixel_side(z_w, y_w, PANEL_W, PANEL_H)
        pts.append((offset_x + px, offset_y + py))

    for i in range(1, len(pts)):
        alpha = i / len(pts)
        c = tuple(int(c * alpha) for c in color)
        cv2.line(canvas, pts[i - 1], pts[i], c, 2, cv2.LINE_AA)


# ── Sidebar & overlay ──────────────────────────────────────────────────────────

def draw_sidebar(canvas, fall_prob, pred, frame_idx, total_frames,
                 window_size, step, gt_label, threshold, fps_actual):
    """Sidebar kanan: info + confidence bar."""
    sx = PANEL_W * 2 + 4   # x start sidebar
    sw = SIDE_W
    sh = WIN_H

    # Background sidebar
    cv2.rectangle(canvas, (sx, 0), (sx + sw, sh), (15, 15, 25), -1)
    cv2.line(canvas, (sx, 0), (sx, sh), (60, 60, 80), 1)

    y = 20
    def txt(s, col=(220, 220, 220), sz=0.5, thick=1):
        nonlocal y
        cv2.putText(canvas, s, (sx + 10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    sz, col, thick, cv2.LINE_AA)
        y += int(sz * 38)

    txt("FALL DETECTION", (180, 220, 255), 0.6, 2)
    txt("NTU 25-joint", (120, 150, 200), 0.45)
    y += 5
    cv2.line(canvas, (sx + 10, y), (sx + sw - 10, y), (60, 60, 80), 1)
    y += 12

    # Status utama
    status_col = FALL_COLOR if pred == 1 else SAFE_COLOR
    status_str = "  FALL  " if pred == 1 else "NOT FALL"
    cv2.rectangle(canvas,
                  (sx + 10, y), (sx + sw - 10, y + 45),
                  status_col, -1)
    cv2.putText(canvas, status_str,
                (sx + 18, y + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    y += 55

    # Confidence bar
    txt("P(fall): {:.3f}".format(fall_prob), (200, 200, 200), 0.5)
    bar_x = sx + 10
    bar_w = sw - 20
    bar_h = 18
    # Background bar
    cv2.rectangle(canvas, (bar_x, y), (bar_x + bar_w, y + bar_h),
                  (50, 50, 60), -1)
    # Filled bar (gradient hijau→merah)
    filled = int(fall_prob * bar_w)
    r = int(fall_prob * 255)
    g = int((1 - fall_prob) * 200)
    cv2.rectangle(canvas, (bar_x, y), (bar_x + filled, y + bar_h),
                  (30, g, r), -1)
    # Threshold line
    thr_x = bar_x + int(threshold * bar_w)
    cv2.line(canvas, (thr_x, y - 3), (thr_x, y + bar_h + 3), (255, 255, 100), 2)
    cv2.rectangle(canvas, (bar_x, y), (bar_x + bar_w, y + bar_h),
                  (80, 80, 100), 1)
    y += bar_h + 14

    cv2.line(canvas, (sx + 10, y), (sx + sw - 10, y), (60, 60, 80), 1)
    y += 10

    # Info frame
    txt("Frame   : {}/{}".format(frame_idx + 1, total_frames), sz=0.45)
    txt("Window  : {}  Step: {}".format(window_size, step), sz=0.45)
    txt("FPS     : {:.1f}".format(fps_actual), sz=0.45)
    txt("Thr     : {:.2f}".format(threshold), sz=0.45)

    # Ground truth
    if gt_label is not None:
        y += 5
        cv2.line(canvas, (sx + 10, y), (sx + sw - 10, y), (60, 60, 80), 1)
        y += 10
        gt_str = "FALL" if gt_label == 1 else "NOT FALL"
        correct = (pred == gt_label)
        gt_col = (100, 255, 100) if correct else (100, 100, 255)
        txt("GT: {}".format(gt_str), gt_col, sz=0.5)
        txt("{}".format("BENAR" if correct else "SALAH"), gt_col, sz=0.5, thick=2)

    # Kontrol
    y = sh - 130
    cv2.line(canvas, (sx + 10, y), (sx + sw - 10, y), (60, 60, 80), 1)
    y += 10
    txt("Kontrol:", (140, 140, 160), 0.42)
    txt("SPACE  pause/resume", (120, 120, 140), 0.4)
    txt("+/-    speed", (120, 120, 140), 0.4)
    txt("T      trajektori", (120, 120, 140), 0.4)
    txt("S      screenshot", (120, 120, 140), 0.4)
    txt("Q/ESC  keluar", (120, 120, 140), 0.4)


def draw_panel_label(canvas, text, x, y, w):
    """Label di atas panel."""
    cv2.putText(canvas, text, (x + w // 2 - len(text) * 5, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 180, 200), 1, cv2.LINE_AA)


def draw_fall_alert(canvas, fall_prob, threshold):
    """Overlay merah berkedip saat fall terdeteksi."""
    if fall_prob < threshold:
        return
    intensity = int((fall_prob - threshold) / (1.0 - threshold) * 80)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (PANEL_W * 2 + 4, WIN_H),
                  (0, 0, 200), -1)
    cv2.addWeighted(overlay, intensity / 255, canvas,
                    1 - intensity / 255, 0, canvas)
    # Border merah tebal
    cv2.rectangle(canvas, (2, 2), (PANEL_W * 2, WIN_H - 2),
                  (0, 0, 255), 3)
    # Teks alert
    cv2.putText(canvas, "! FALL DETECTED !",
                (PANEL_W - 120, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 3, cv2.LINE_AA)


def draw_status_bar(canvas, frame_idx, total_frames, paused, fps_target):
    """Bar bawah: progress + info."""
    bar_y = PANEL_H + 4
    bar_h = WIN_H - bar_y
    cv2.rectangle(canvas, (0, bar_y), (PANEL_W * 2 + 4, WIN_H),
                  (12, 12, 22), -1)
    cv2.line(canvas, (0, bar_y), (PANEL_W * 2 + 4, bar_y), (60, 60, 80), 1)

    # Progress bar
    bar_x = 10
    bar_w = PANEL_W * 2 - 20
    prog  = frame_idx / max(total_frames - 1, 1)
    cv2.rectangle(canvas, (bar_x, bar_y + 20),
                  (bar_x + bar_w, bar_y + 35),
                  (40, 40, 55), -1)
    cv2.rectangle(canvas, (bar_x, bar_y + 20),
                  (bar_x + int(prog * bar_w), bar_y + 35),
                  (80, 160, 220), -1)
    cv2.rectangle(canvas, (bar_x, bar_y + 20),
                  (bar_x + bar_w, bar_y + 35),
                  (70, 70, 90), 1)

    # Status teks
    pause_str = "[PAUSED] " if paused else ""
    status = "{}Frame {}/{} | FPS target: {}".format(
        pause_str, frame_idx + 1, total_frames, fps_target)
    cv2.putText(canvas, status,
                (bar_x, bar_y + 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (160, 160, 180), 1, cv2.LINE_AA)


# ── Main loop ──────────────────────────────────────────────────────────────────

def get_ground_truth(filepath):
    stem = Path(filepath).stem
    try:
        a_parts = [p for p in stem.split("A") if len(p) >= 3 and p[:3].isdigit()]
        act = int(a_parts[-1][:3])
    except (IndexError, ValueError):
        return None, -1
    if act == 43:
        return 1, act
    if act in {8, 9, 27, 42}:
        return 0, act
    return None, act


def run_visualization(args):
    # ── Load skeleton ──────────────────────────────────────────────────────────
    print("Memuat skeleton: {}".format(args.skeleton))
    _, all_frames = read_skeleton_file(args.skeleton)
    total_frames  = len(all_frames)
    gt_label, act_class = get_ground_truth(args.skeleton)

    if not all_frames:
        sys.exit("[ERROR] File skeleton kosong: {}".format(args.skeleton))

    print("  Total frame : {}".format(total_frames))
    gt_str = "A{:03d} → {}".format(
        act_class,
        "FALL" if gt_label == 1 else "NOT FALL" if gt_label == 0 else "unknown"
    )
    print("  Ground truth: {}".format(gt_str))

    # ── Load model (opsional) ──────────────────────────────────────────────────
    model  = None
    device = torch.device("cpu")
    if args.weights and args.config:
        if not Path(args.weights).exists():
            print("[WARN] Weights tidak ditemukan: {}".format(args.weights))
        else:
            print("Memuat model ...")
            device = torch.device(
                "cuda" if torch.cuda.is_available() and args.device != "cpu"
                else "cpu"
            )
            model = load_model_from_config(args.config, args.weights, device)
            print("  Model siap pada {}".format(device))
    else:
        print("[INFO] Tanpa model — hanya visualisasi skeleton")

    # ── State ──────────────────────────────────────────────────────────────────
    window_buf     = deque(maxlen=args.window)
    traj_buf       = deque(maxlen=args.window)   # untuk trajektori SpineBase
    fall_prob      = 0.0
    pred           = 0
    paused         = False
    show_traj      = True
    fps_target     = args.fps
    frame_idx      = 0
    last_infer_idx = -1
    fps_actual     = float(fps_target)
    t_last         = time.perf_counter()

    cv2.namedWindow("Fall Detection — NTU RGB+D 25-joint", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Fall Detection — NTU RGB+D 25-joint", WIN_W, WIN_H)

    print("\nMulai visualisasi... (SPACE=pause, Q=keluar)")

    screenshot_counter = 0

    while frame_idx < total_frames:
        # ── Keyboard ──────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
        elif key in (ord("+"), ord("=")):
            fps_target = min(fps_target + 5, 120)
        elif key == ord("-"):
            fps_target = max(fps_target - 5, 1)
        elif key == ord("t"):
            show_traj = not show_traj
            traj_buf.clear()
        elif key == ord("s"):
            canvas_save = canvas.copy() if "canvas" in dir() else None
            if canvas_save is not None:
                fn = "screenshot_{:03d}.png".format(screenshot_counter)
                cv2.imwrite(fn, canvas_save)
                print("Screenshot disimpan: {}".format(fn))
                screenshot_counter += 1

        if paused:
            time.sleep(0.02)
            continue

        # ── Ambil frame ───────────────────────────────────────────────────────
        frame_bodies = all_frames[frame_idx]
        if frame_bodies:
            joints = frame_bodies[0]["joints"]   # (25, 3) absolut
        else:
            joints = np.zeros((NUM_JOINTS, 3), np.float32)

        window_buf.append(joints.copy())
        if show_traj:
            traj_buf.append(joints.copy())

        # ── Inferensi sliding window ───────────────────────────────────────────
        if (model is not None and
                len(window_buf) >= min(30, args.window) and
                (frame_idx - last_infer_idx) >= args.step):
            x = window_to_input(list(window_buf), args.window).to(device)
            with torch.no_grad():
                logits = model(x)
            probs     = torch.softmax(logits, dim=1)[0].cpu().numpy()
            fall_prob = float(probs[1])
            pred      = 1 if fall_prob >= args.threshold else 0
            last_infer_idx = frame_idx

        # ── Render canvas ─────────────────────────────────────────────────────
        canvas = np.full((WIN_H, WIN_W, 3), BG_COLOR, dtype=np.uint8)

        # Panel dividers
        cv2.line(canvas, (PANEL_W + 2, 0), (PANEL_W + 2, PANEL_H), (50, 50, 70), 2)
        cv2.line(canvas, (PANEL_W * 2 + 4, 0), (PANEL_W * 2 + 4, WIN_H), (50, 50, 70), 1)

        # Grid
        draw_grid(canvas, 0,          0, PANEL_W, PANEL_H)
        draw_grid(canvas, PANEL_W + 4, 0, PANEL_W, PANEL_H)

        # Label panel
        draw_panel_label(canvas, "FRONT VIEW  (X-Y)", 0,          0, PANEL_W)
        draw_panel_label(canvas, "SIDE VIEW  (Z-Y)",  PANEL_W + 4, 0, PANEL_W)

        # Trajektori SpineBase
        if show_traj and len(traj_buf) > 1:
            draw_trajectory(canvas, list(traj_buf), 0,           8, view="front")
            draw_trajectory(canvas, list(traj_buf), PANEL_W + 4, 8, view="side")

        # Skeleton (frame saat ini)
        is_valid = np.any(joints != 0)
        draw_skeleton(canvas, joints, 0,           8, view="front", is_valid=is_valid)
        draw_skeleton(canvas, joints, PANEL_W + 4, 8, view="side",  is_valid=is_valid)

        # Overlay alert fall
        draw_fall_alert(canvas, fall_prob, args.threshold)

        # Sidebar
        draw_sidebar(canvas, fall_prob, pred,
                     frame_idx, total_frames,
                     args.window, args.step,
                     gt_label, args.threshold, fps_actual)

        # Status bar bawah
        draw_status_bar(canvas, frame_idx, total_frames, paused, fps_target)

        # ── Tampilkan ────────────────────────────────────────────────────────
        cv2.imshow("Fall Detection — NTU RGB+D 25-joint", canvas)

        # ── Timing ──────────────────────────────────────────────────────────
        t_now = time.perf_counter()
        elapsed = t_now - t_last
        target  = 1.0 / fps_target
        sleep   = max(0.0, target - elapsed)
        if sleep > 0:
            time.sleep(sleep)

        fps_actual = 1.0 / max(time.perf_counter() - t_last, 1e-9)
        t_last = time.perf_counter()
        frame_idx += 1

    cv2.destroyAllWindows()
    print("\nVisualisasi selesai.")
    if model is not None:
        print("Prediksi akhir  : {} (P(fall)={:.4f})".format(
            "FALL" if pred == 1 else "NOT FALL", fall_prob))
        if gt_label is not None:
            correct = (pred == gt_label)
            print("Ground truth    : {}  → {}".format(
                "FALL" if gt_label == 1 else "NOT FALL",
                "BENAR" if correct else "SALAH"))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Visualisasi real-time skeleton NTU RGB+D dengan fall detection")
    ap.add_argument("--skeleton",   default=SKELETON_PATH,
                    help="Path file .skeleton NTU RGB+D")
    ap.add_argument("--weights",    default=WEIGHTS_PATH,
                    help="Path model weights .pt (opsional)")
    ap.add_argument("--config",     default=CONFIG_PATH,
                    help="Path config YAML")
    ap.add_argument("--window",     type=int, default=150,
                    help="Ukuran sliding window (frame)")
    ap.add_argument("--step",       type=int, default=10,
                    help="Jalankan inferensi setiap N frame")
    ap.add_argument("--threshold",  type=float, default=0.5,
                    help="Threshold P(fall)")
    ap.add_argument("--fps",        type=float, default=20.0,
                    help="Target FPS simulasi (default 20)")
    ap.add_argument("--device",     default="auto",
                    choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    skel_path = Path(args.skeleton)
    if not skel_path.exists():
        sys.exit("[ERROR] Path tidak ditemukan: {}".format(args.skeleton))

    # Jika path adalah direktori, pilih file .skeleton dari dalamnya
    if skel_path.is_dir():
        all_files = sorted(skel_path.rglob("*.skeleton"))
        if not all_files:
            sys.exit("[ERROR] Tidak ada file .skeleton di: {}".format(args.skeleton))

        # Filter ke kelas yang relevan (A043, A008, A009, A027, A042)
        VALID_ACTS = {8, 9, 27, 42, 43}
        relevant = []
        for fp in all_files:
            try:
                a_parts = [p for p in fp.stem.split("A")
                           if len(p) >= 3 and p[:3].isdigit()]
                act = int(a_parts[-1][:3])
                if act in VALID_ACTS:
                    relevant.append((act, fp))
            except (IndexError, ValueError):
                pass

        if not relevant:
            # Fallback: ambil file apapun
            relevant = [(-1, f) for f in all_files]

        # Tampilkan daftar file (max 30) dan minta pilihan
        print("\nFolder berisi {} file .skeleton ({} relevan untuk fall detection)".format(
            len(all_files), len(relevant)))
        print("\nDaftar file (ketik nomor untuk memilih):")
        show = relevant[:30]
        for i, (act, fp) in enumerate(show):
            tag = ""
            if act == 43:
                tag = " [FALL]"
            elif act in {8, 9, 27, 42}:
                tag = " [not-fall]"
            print("  {:2d}. {}{}".format(i, fp.name, tag))
        if len(relevant) > 30:
            print("  ... ({} file lainnya)".format(len(relevant) - 30))

        try:
            choice = input("\nPilih nomor file [0]: ").strip()
            idx = int(choice) if choice else 0
            idx = max(0, min(idx, len(show) - 1))
        except (ValueError, EOFError):
            idx = 0

        args.skeleton = str(show[idx][1])
        print("Dipilih: {}\n".format(Path(args.skeleton).name))

    run_visualization(args)


if __name__ == "__main__":
    main()
