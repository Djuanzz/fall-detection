"""
realtime_skeleton_viz_multi.py
==============================
Visualisasi real-time gabungan N file .skeleton NTU RGB+D sebagai
satu kejadian runtut, lengkap dengan fall detection overlay.

Tampilan:
  - Dua panel: Front View (X-Y) dan Side View (Z-Y)
  - Skeleton 25-joint berwarna per segmen tubuh
  - Sidebar: status prediksi, confidence bar, info klip
  - Timeline bawah: semua klip (merah=fall, biru=not-fall) + posisi sekarang
  - Judul klip muncul saat transisi antar video
  - Sliding window inference lintas klip (tidak direset)

Kontrol:
  SPACE   : pause / resume
  Q / ESC : keluar
  +/-     : percepat / perlambat
  T       : toggle trajektori SpineBase
  N       : loncat ke klip berikutnya
  S       : screenshot

Cara pakai:
    # Otomatis pilih 15 klip dari folder (mix fall + not-fall)
    python scripts/realtime_skeleton_viz_multi.py \\
        --skeleton_dir  /path/to/nturgbd_skeletons \\
        --weights       work_dir/fall_ntu25_balanced/runs-82-15498.pt \\
        --config        config/fall-detection-ntu/balanced.yaml \\
        --n_clips 15

    # Atau daftar file manual
    python scripts/realtime_skeleton_viz_multi.py \\
        --files file1.skeleton file2.skeleton ... \\
        --weights work_dir/...
"""

SKELETON_DIR = "e:\\000 tugasakhir\\03 code\\block-gcn-yolo\\data\\nturgbd_raw\\nturgb+d_skeletons"
WEIGHTS_PATH = "work_dir/fall_ntu25_balanced/runs-82-15498.pt"
CONFIG_PATH  = "config/fall-detection-ntu/balanced.yaml"

import argparse
import importlib
import random
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.prepare_dataset_ntu25 import (
    read_skeleton_file,
    normalize_skeleton,
    CHANNELS,
    NUM_JOINTS,
)

# ── Layout ─────────────────────────────────────────────────────────────────────

PANEL_W  = 580
PANEL_H  = 520
SIDE_W   = 280
TBAR_H   = 90       # tinggi timeline bar
WIN_W    = PANEL_W * 2 + SIDE_W + 4
WIN_H    = PANEL_H + TBAR_H + 4

# Warna BGR
C_SPINE   = (80,  230,  80)
C_LARM    = (255, 140,  60)
C_RARM    = (60,  140, 255)
C_LLEG    = (210, 200, 100)
C_RLEG    = (100, 200, 210)
C_FINGER  = (180, 180, 180)
C_JOINT   = (255, 255, 255)
C_SPINE_J = (80,  230,  80)
C_HEAD_J  = (255, 200, 100)
BG_COLOR  = (20,  20,   30)
GRID_COL  = (40,  40,   55)
FALL_COL  = (50,  50,  240)
SAFE_COL  = (50,  200,  50)

# Kelas aksi relevan
FALL_ACTS     = {43}
NOT_FALL_ACTS = {8, 9, 27, 42}
ALL_ACTS      = FALL_ACTS | NOT_FALL_ACTS

ACT_NAMES = {
    8:  "Sitting down",
    9:  "Standing up",
    27: "Jump up",
    42: "Staggering",
    43: "FALL DOWN",
}

BONES = [
    ((0, 1), C_SPINE), ((1, 20), C_SPINE), ((20, 2), C_SPINE), ((2, 3), C_SPINE),
    ((20, 4), C_LARM), ((4, 5), C_LARM), ((5, 6), C_LARM), ((6, 7), C_LARM),
    ((7, 21), C_FINGER), ((6, 22), C_FINGER),
    ((20, 8), C_RARM), ((8, 9), C_RARM), ((9, 10), C_RARM), ((10, 11), C_RARM),
    ((11, 23), C_FINGER), ((10, 24), C_FINGER),
    ((0, 12), C_LLEG), ((12, 13), C_LLEG), ((13, 14), C_LLEG), ((14, 15), C_LLEG),
    ((0, 16), C_RLEG), ((16, 17), C_RLEG), ((17, 18), C_RLEG), ((18, 19), C_RLEG),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_action_class(filepath):
    stem = Path(filepath).stem
    try:
        parts = [p for p in stem.split("A") if len(p) >= 3 and p[:3].isdigit()]
        return int(parts[-1][:3])
    except (IndexError, ValueError):
        return -1


def act_to_label(act):
    if act in FALL_ACTS:     return 1
    if act in NOT_FALL_ACTS: return 0
    return None


def clip_color(label):
    """Warna klip untuk timeline: merah=fall, biru=not-fall, abu=unknown."""
    if label == 1: return (60, 60, 220)
    if label == 0: return (60, 180, 60)
    return (100, 100, 100)


# ── Pemilihan file dari direktori ──────────────────────────────────────────────

def pick_clips_from_dir(skeleton_dir, n_clips, mix_mode, seed):
    """
    Pilih n_clips file dari direktori dengan strategi mix_mode:
      'narrative' : urutan not-fall → fall → not-fall (lebih menarik untuk demo)
      'random'    : acak dari semua kelas relevan
    """
    skel_dir = Path(skeleton_dir)
    all_files = sorted(skel_dir.rglob("*.skeleton"))
    if not all_files:
        sys.exit("[ERROR] Tidak ada file .skeleton di: {}".format(skeleton_dir))

    by_class = {1: [], 0: []}
    for fp in all_files:
        act = get_action_class(fp.name)
        lbl = act_to_label(act)
        if lbl is not None:
            by_class[lbl].append(fp)

    n_fall     = max(1, n_clips // 4)          # ~25% fall
    n_not_fall = n_clips - n_fall

    rng = random.Random(seed)
    fall_pool     = rng.sample(by_class[1], min(n_fall,     len(by_class[1])))
    notfall_pool  = rng.sample(by_class[0], min(n_not_fall, len(by_class[0])))

    if mix_mode == "narrative":
        # Urutan: beberapa not-fall → fall → not-fall → fall → not-fall ...
        result = []
        fi, ni = 0, 0
        toggle = 0
        while len(result) < n_clips:
            # 3-4 not-fall, lalu 1 fall
            batch_nf = rng.randint(2, 4)
            for _ in range(batch_nf):
                if ni < len(notfall_pool) and len(result) < n_clips:
                    result.append(notfall_pool[ni])
                    ni += 1
            if fi < len(fall_pool) and len(result) < n_clips:
                result.append(fall_pool[fi])
                fi += 1
            if fi >= len(fall_pool) and ni >= len(notfall_pool):
                break
    else:
        combined = fall_pool + notfall_pool
        rng.shuffle(combined)
        result = combined[:n_clips]

    print("Dipilih {} klip:".format(len(result)))
    for i, fp in enumerate(result):
        act = get_action_class(fp.name)
        tag = " [FALL]" if act in FALL_ACTS else " [not-fall]"
        print("  {:2d}. {}{}".format(i + 1, fp.name, tag))
    return [str(fp) for fp in result]


# ── Pemuatan & penggabungan multi-klip ────────────────────────────────────────

def load_and_concat(file_list):
    """
    Muat semua .skeleton, gabung jadi satu sequence panjang.
    Return:
      combined  : np.ndarray (TotalFrames, 25, 3)  koordinat absolut
      segments  : list of dict per klip
    """
    segments = []
    all_joints = []
    cursor = 0

    for fp in file_list:
        try:
            _, frames = read_skeleton_file(fp)
        except Exception as e:
            print("[WARN] Gagal baca {}: {}".format(Path(fp).name, e))
            continue

        act   = get_action_class(fp)
        label = act_to_label(act)
        n     = len(frames)

        clip_joints = np.zeros((n, NUM_JOINTS, 3), np.float32)
        for t, frame_bodies in enumerate(frames):
            if frame_bodies:
                clip_joints[t] = frame_bodies[0]["joints"]

        all_joints.append(clip_joints)
        segments.append({
            "filepath":   fp,
            "filename":   Path(fp).name,
            "act":        act,
            "label":      label,
            "act_name":   ACT_NAMES.get(act, "A{:03d}".format(act)),
            "start":      cursor,
            "length":     n,
            "end":        cursor + n,
        })
        cursor += n

    combined = np.concatenate(all_joints, axis=0) if all_joints else np.zeros((1, NUM_JOINTS, 3), np.float32)
    return combined, segments


def find_segment(segments, frame_idx):
    """Cari segmen yang mengandung frame_idx."""
    for seg in segments:
        if seg["start"] <= frame_idx < seg["end"]:
            return seg
    return segments[-1]


# ── Model ──────────────────────────────────────────────────────────────────────

def load_model_from_config(config_path, weights_path, device):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cls_path = cfg.get("model", "model.BlockGCN.Model")
    parts = cls_path.split(".")
    mod = importlib.import_module(".".join(parts[:-1]))
    model = getattr(mod, parts[-1])(**cfg["model_args"]).to(device)
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


def window_to_input(frames_raw, window_size):
    seq = np.stack(frames_raw, axis=0)
    seq_norm = normalize_skeleton(seq)
    T = seq_norm.shape[0]
    if T < window_size:
        pad = np.zeros((window_size - T, NUM_JOINTS, CHANNELS), np.float32)
        seq_norm = np.concatenate([seq_norm, pad], axis=0)
    else:
        seq_norm = seq_norm[-window_size:]
    t = seq_norm.transpose(2, 0, 1)[:, :, :, np.newaxis]
    return torch.tensor(t[np.newaxis], dtype=torch.float32)


# ── Proyeksi world → pixel ─────────────────────────────────────────────────────

def w2p_front(x, y, pw, ph, xr=(-1.2, 1.2), yr=(-0.3, 2.2)):
    px = int((x - xr[0]) / (xr[1] - xr[0]) * pw)
    py = int((1.0 - (y - yr[0]) / (yr[1] - yr[0])) * ph)
    return np.clip(px, 0, pw - 1), np.clip(py, 0, ph - 1)


def w2p_side(z, y, pw, ph, zr=(0.3, 4.5), yr=(-0.3, 2.2)):
    px = int((z - zr[0]) / (zr[1] - zr[0]) * pw)
    py = int((1.0 - (y - yr[0]) / (yr[1] - yr[0])) * ph)
    return np.clip(px, 0, pw - 1), np.clip(py, 0, ph - 1)


# ── Render functions ───────────────────────────────────────────────────────────

def draw_grid(canvas, ox, oy, pw, ph, n=8):
    for i in range(n + 1):
        x = ox + i * pw // n
        cv2.line(canvas, (x, oy), (x, oy + ph), GRID_COL, 1, cv2.LINE_AA)
    step = pw // n
    y = oy
    while y <= oy + ph:
        cv2.line(canvas, (ox, y), (ox + pw, y), GRID_COL, 1, cv2.LINE_AA)
        y += step


def draw_skeleton(canvas, joints, ox, oy, view="front"):
    pw, ph = PANEL_W, PANEL_H

    def to_px(j):
        xw, yw, zw = joints[j]
        if view == "front":
            return w2p_front(xw, yw, pw, ph)
        return w2p_side(zw, yw, pw, ph)

    for (j1, j2), color in BONES:
        if np.all(joints[j1] == 0) or np.all(joints[j2] == 0):
            continue
        p1 = (ox + to_px(j1)[0], oy + to_px(j1)[1])
        p2 = (ox + to_px(j2)[0], oy + to_px(j2)[1])
        cv2.line(canvas, p1, p2, color, 3, cv2.LINE_AA)

    for j in range(NUM_JOINTS):
        if np.all(joints[j] == 0):
            continue
        px, py = to_px(j)
        r = 7 if j == 3 else 5 if j in (0, 1, 2, 20) else 4
        c = C_HEAD_J if j == 3 else C_SPINE_J if j in (0, 1, 2, 20) else C_JOINT
        cv2.circle(canvas, (ox + px, oy + py), r, c, -1, cv2.LINE_AA)
        cv2.circle(canvas, (ox + px, oy + py), r + 1, (0, 0, 0), 1, cv2.LINE_AA)


def draw_trajectory(canvas, traj, ox, oy, view="front"):
    pts = []
    for joints in traj:
        xw, yw, zw = joints[0]
        if view == "front":
            px, py = w2p_front(xw, yw, PANEL_W, PANEL_H)
        else:
            px, py = w2p_side(zw, yw, PANEL_W, PANEL_H)
        pts.append((ox + px, oy + py))
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        c = (int(200 * alpha), int(200 * alpha), int(50 * alpha))
        cv2.line(canvas, pts[i - 1], pts[i], c, 2, cv2.LINE_AA)


def draw_clip_title(canvas, seg, frame_local, flash=False):
    """Overlay judul klip di tengah atas saat transisi."""
    label_col = (100, 80, 255) if seg["label"] == 1 else (80, 200, 80)
    if flash:
        ov = canvas.copy()
        cv2.rectangle(ov, (0, 0), (PANEL_W * 2 + 4, 70), (10, 10, 20), -1)
        cv2.addWeighted(ov, 0.7, canvas, 0.3, 0, canvas)

    title = "Klip {}  |  {}".format(
        seg.get("clip_idx", "?") + 1, seg["act_name"])
    ts = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    tx = (PANEL_W * 2 + 4) // 2 - ts[0] // 2
    cv2.putText(canvas, title, (tx, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_col, 2, cv2.LINE_AA)
    sub = "{} / {} frame  ({})".format(
        frame_local + 1, seg["length"], seg["filename"])
    ts2 = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
    cv2.putText(canvas, sub,
                ((PANEL_W * 2 + 4) // 2 - ts2[0] // 2, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 170), 1, cv2.LINE_AA)


def draw_sidebar(canvas, fall_prob, pred, seg, clip_idx, n_clips,
                 total_frames, frame_idx, window_size, threshold, fps):
    sx = PANEL_W * 2 + 4
    sw = SIDE_W

    cv2.rectangle(canvas, (sx, 0), (sx + sw, WIN_H), (15, 15, 25), -1)
    cv2.line(canvas, (sx, 0), (sx, WIN_H), (60, 60, 80), 1)

    y = 18
    def txt(s, col=(220, 220, 220), sz=0.48, th=1):
        nonlocal y
        cv2.putText(canvas, s, (sx + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, sz, col, th, cv2.LINE_AA)
        y += int(sz * 40)

    txt("FALL DETECTION", (180, 220, 255), 0.58, 2)
    txt("NTU RGB+D 25-joint", (100, 140, 190), 0.4)
    y += 4
    cv2.line(canvas, (sx + 8, y), (sx + sw - 8, y), (60, 60, 80), 1)
    y += 10

    # Status box
    sc = FALL_COL if pred == 1 else SAFE_COL
    ss = "  FALL  " if pred == 1 else "NOT FALL"
    cv2.rectangle(canvas, (sx + 8, y), (sx + sw - 8, y + 42), sc, -1)
    cv2.putText(canvas, ss, (sx + 16, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    y += 52

    txt("P(fall)  {:.4f}".format(fall_prob), sz=0.46)
    bx, bw, bh = sx + 8, sw - 16, 16
    cv2.rectangle(canvas, (bx, y), (bx + bw, y + bh), (45, 45, 60), -1)
    filled = int(fall_prob * bw)
    r = int(fall_prob * 255); g = int((1 - fall_prob) * 200)
    cv2.rectangle(canvas, (bx, y), (bx + filled, y + bh), (30, g, r), -1)
    tx_ = bx + int(threshold * bw)
    cv2.line(canvas, (tx_, y - 3), (tx_, y + bh + 3), (255, 230, 50), 2)
    cv2.rectangle(canvas, (bx, y), (bx + bw, y + bh), (80, 80, 100), 1)
    y += bh + 12

    cv2.line(canvas, (sx + 8, y), (sx + sw - 8, y), (60, 60, 80), 1)
    y += 8

    # Info klip
    txt("Klip  {} / {}".format(clip_idx + 1, n_clips), sz=0.45)
    txt("Aksi  {}".format(seg["act_name"]), sz=0.43,
        col=(100, 80, 255) if seg["label"] == 1 else (80, 200, 80))
    txt("Frame {}/{}".format(frame_idx + 1, total_frames), sz=0.43)
    txt("FPS   {:.1f}".format(fps), sz=0.43)

    # Ground truth klip ini
    if seg["label"] is not None:
        y += 4
        cv2.line(canvas, (sx + 8, y), (sx + sw - 8, y), (60, 60, 80), 1)
        y += 8
        gt_str = "FALL" if seg["label"] == 1 else "NOT FALL"
        gt_col = (100, 80, 255) if seg["label"] == 1 else (80, 200, 80)
        txt("GT    {}".format(gt_str), gt_col, 0.47, 2)
        res = pred == seg["label"]
        txt("      {}".format("BENAR" if res else "SALAH"),
            (100, 230, 100) if res else (100, 100, 230), 0.47)

    # Kontrol
    y = WIN_H - 115
    cv2.line(canvas, (sx + 8, y), (sx + sw - 8, y), (60, 60, 80), 1)
    y += 8
    txt("Kontrol:", (140, 140, 160), 0.4)
    txt("SPACE pause  +/- speed", (110, 110, 130), 0.38)
    txt("T traj  N next clip", (110, 110, 130), 0.38)
    txt("S screenshot  Q quit", (110, 110, 130), 0.38)


def draw_timeline(canvas, segments, frame_idx, total_frames, fall_prob_history):
    """
    Timeline bar di bagian bawah canvas.
    Setiap klip ditampilkan sebagai blok berwarna.
    Posisi saat ini ditandai garis putih.
    """
    ty   = PANEL_H + 4
    tw   = PANEL_W * 2 + 4
    th   = TBAR_H

    cv2.rectangle(canvas, (0, ty), (tw, ty + th), (12, 12, 22), -1)
    cv2.line(canvas, (0, ty), (tw, ty), (55, 55, 75), 1)

    # Label
    cv2.putText(canvas, "Timeline", (8, ty + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 160), 1, cv2.LINE_AA)

    # Klip blocks
    bar_x, bar_y = 8, ty + 22
    bar_w, bar_h = tw - 16, 24

    for seg in segments:
        x1 = bar_x + int(seg["start"] / total_frames * bar_w)
        x2 = bar_x + int(seg["end"]   / total_frames * bar_w)
        col = (50, 50, 180) if seg["label"] == 1 else (50, 150, 50)
        cv2.rectangle(canvas, (x1, bar_y), (x2 - 1, bar_y + bar_h), col, -1)
        cv2.rectangle(canvas, (x1, bar_y), (x2 - 1, bar_y + bar_h), (0, 0, 0), 1)
        # Nomor klip jika cukup lebar
        if x2 - x1 > 20:
            cv2.putText(canvas,
                        str(seg["clip_idx"] + 1),
                        (x1 + 3, bar_y + bar_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (230, 230, 230), 1)

    # Fall probability mini-graph
    if fall_prob_history:
        gx, gy = bar_x, bar_y + bar_h + 4
        gh = 22
        # background
        cv2.rectangle(canvas, (gx, gy), (gx + bar_w, gy + gh), (25, 25, 35), -1)
        pts = []
        for i, p in enumerate(fall_prob_history):
            x = gx + int(i / max(len(fall_prob_history) - 1, 1) * bar_w)
            y = gy + gh - int(p * gh)
            pts.append((x, y))
        for i in range(1, len(pts)):
            r = int(fall_prob_history[i] * 255)
            g = int((1 - fall_prob_history[i]) * 180)
            cv2.line(canvas, pts[i - 1], pts[i], (30, g, r), 2, cv2.LINE_AA)
        # Label
        cv2.putText(canvas, "P(fall)", (gx + 2, gy + gh - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 140, 160), 1)

    # Posisi saat ini
    cx = bar_x + int(frame_idx / max(total_frames - 1, 1) * bar_w)
    cv2.line(canvas, (cx, bar_y - 2), (cx, bar_y + bar_h + 26 + 4),
             (255, 255, 255), 2, cv2.LINE_AA)

    # Threshold line pada mini-graph
    if fall_prob_history:
        thr_y_px = bar_y + bar_h + 4 + 22 - int(0.5 * 22)
        cv2.line(canvas, (bar_x, thr_y_px), (bar_x + bar_w, thr_y_px),
                 (200, 200, 50), 1, cv2.LINE_AA)


def draw_fall_border(canvas, fall_prob, threshold):
    if fall_prob < threshold:
        return
    intensity = int((fall_prob - threshold) / max(1.0 - threshold, 1e-6) * 90)
    ov = canvas.copy()
    cv2.rectangle(ov, (0, 0), (PANEL_W * 2 + 4, PANEL_H + 4), (0, 0, 200), -1)
    cv2.addWeighted(ov, intensity / 255.0, canvas, 1 - intensity / 255.0, 0, canvas)
    cv2.rectangle(canvas, (2, 2), (PANEL_W * 2 + 2, PANEL_H + 2), (0, 0, 255), 4)


# ── Main visualisasi ───────────────────────────────────────────────────────────

def run_multi_viz(args, file_list):
    print("\nMemuat {} klip skeleton...".format(len(file_list)))
    combined, segments = load_and_concat(file_list)
    total_frames = len(combined)

    for i, seg in enumerate(segments):
        seg["clip_idx"] = i

    if total_frames == 0:
        sys.exit("[ERROR] Tidak ada frame yang berhasil dimuat.")

    print("Total frame gabungan: {}  ({:.1f} detik pada 30fps)".format(
        total_frames, total_frames / 30.0))

    # Load model
    model = None
    device = torch.device("cpu")
    if args.weights and Path(args.weights).exists():
        device = torch.device(
            "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
        print("Memuat model pada {}...".format(device))
        model = load_model_from_config(args.config, args.weights, device)
        print("Model siap.")
    else:
        print("[INFO] Tanpa model — hanya visualisasi skeleton.")

    # State
    window_buf       = deque(maxlen=args.window)
    traj_buf         = deque(maxlen=60)
    fall_prob        = 0.0
    pred             = 0
    paused           = False
    show_traj        = True
    fps_target       = args.fps
    fps_actual       = float(fps_target)
    frame_idx        = 0
    last_infer_idx   = -1
    prev_seg_idx     = -1
    show_title_until = 0       # frame_idx sampai judul klip ditampilkan
    screenshot_n     = 0
    fall_prob_hist   = []      # riwayat fall_prob sepanjang sequence

    cv2.namedWindow("Fall Detection — Multi Klip", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Fall Detection — Multi Klip", WIN_W, WIN_H)

    t_last = time.perf_counter()

    while frame_idx < total_frames:
        # Keyboard
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
        elif key == ord("n"):
            # Loncat ke klip berikutnya
            seg_now = find_segment(segments, frame_idx)
            nxt_idx = seg_now["clip_idx"] + 1
            if nxt_idx < len(segments):
                frame_idx = segments[nxt_idx]["start"]
                window_buf.clear()
                traj_buf.clear()
        elif key == ord("s"):
            fn = "multi_screenshot_{:03d}.png".format(screenshot_n)
            if "canvas" in dir():
                cv2.imwrite(fn, canvas)
                print("Screenshot: {}".format(fn))
                screenshot_n += 1

        if paused:
            time.sleep(0.02)
            # Tetap refresh layar saat pause
            if "canvas" in dir():
                cv2.imshow("Fall Detection — Multi Klip", canvas)
            continue

        # Data frame
        joints = combined[frame_idx].copy()   # (25, 3)
        window_buf.append(joints)
        if show_traj:
            traj_buf.append(joints)

        # Segment saat ini
        seg = find_segment(segments, frame_idx)
        cur_seg_idx = seg["clip_idx"]
        frame_local = frame_idx - seg["start"]

        # Deteksi transisi klip baru
        if cur_seg_idx != prev_seg_idx:
            show_title_until = frame_idx + 45   # tampilkan judul 45 frame
            prev_seg_idx = cur_seg_idx
            traj_buf.clear()

        # Inferensi
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

        # Catat riwayat untuk mini-graph
        fall_prob_hist.append(fall_prob)
        # Subsample agar tidak terlalu panjang
        max_hist = PANEL_W * 2
        if len(fall_prob_hist) > max_hist:
            fall_prob_hist = fall_prob_hist[::2]

        # ── Render ─────────────────────────────────────────────────────────────
        canvas = np.full((WIN_H, WIN_W, 3), BG_COLOR, dtype=np.uint8)

        # Divider vertikal
        cv2.line(canvas, (PANEL_W + 2, 0), (PANEL_W + 2, PANEL_H + 4), (50, 50, 70), 2)
        cv2.line(canvas, (PANEL_W * 2 + 4, 0), (PANEL_W * 2 + 4, WIN_H), (50, 50, 70), 1)

        # Grid
        draw_grid(canvas, 0,           0, PANEL_W, PANEL_H)
        draw_grid(canvas, PANEL_W + 4, 0, PANEL_W, PANEL_H)

        # Label view
        cv2.putText(canvas, "FRONT  (X-Y)", (PANEL_W // 2 - 60, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 150, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, "SIDE  (Z-Y)", (PANEL_W + 4 + PANEL_W // 2 - 55, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 150, 180), 1, cv2.LINE_AA)

        # Trajektori
        if show_traj and len(traj_buf) > 1:
            draw_trajectory(canvas, list(traj_buf), 0,           8, "front")
            draw_trajectory(canvas, list(traj_buf), PANEL_W + 4, 8, "side")

        # Skeleton
        is_valid = np.any(joints != 0)
        if is_valid:
            draw_skeleton(canvas, joints, 0,           8, "front")
            draw_skeleton(canvas, joints, PANEL_W + 4, 8, "side")

        # Fall alert border
        draw_fall_border(canvas, fall_prob, args.threshold)

        # Judul klip (muncul saat transisi, atau selalu jika di awal klip)
        flash = (frame_idx == seg["start"])
        draw_clip_title(canvas, seg, frame_local,
                        flash=(frame_idx <= show_title_until))

        # Sidebar
        draw_sidebar(canvas, fall_prob, pred, seg, cur_seg_idx,
                     len(segments), total_frames, frame_idx,
                     args.window, args.threshold, fps_actual)

        # Timeline
        draw_timeline(canvas, segments, frame_idx, total_frames, fall_prob_hist)

        cv2.imshow("Fall Detection — Multi Klip", canvas)

        # Timing
        t_now   = time.perf_counter()
        elapsed = t_now - t_last
        sleep   = max(0.0, 1.0 / fps_target - elapsed)
        if sleep > 0:
            time.sleep(sleep)
        fps_actual = 1.0 / max(time.perf_counter() - t_last, 1e-9)
        t_last = time.perf_counter()
        frame_idx += 1

    cv2.destroyAllWindows()
    print("\nSelesai. {} klip ditampilkan, {} frame total.".format(
        len(segments), total_frames))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Visualisasi multi-klip skeleton NTU RGB+D dengan fall detection")

    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--skeleton_dir", default=SKELETON_DIR,
                     help="Direktori berisi file .skeleton (auto-pilih n_clips)")
    src.add_argument("--files", nargs="+",
                     help="Daftar file .skeleton secara manual")

    ap.add_argument("--n_clips",   type=int, default=15,
                    help="Jumlah klip yang digabung (default 15)")
    ap.add_argument("--mix_mode",  default="narrative",
                    choices=["narrative", "random"],
                    help="Strategi urutan klip: narrative=not-fall..fall.., random=acak")
    ap.add_argument("--seed",      type=int, default=42,
                    help="Random seed untuk reprodusibilitas")
    ap.add_argument("--weights",   default=WEIGHTS_PATH,
                    help="Path model weights .pt")
    ap.add_argument("--config",    default=CONFIG_PATH,
                    help="Path config YAML")
    ap.add_argument("--window",    type=int,   default=150)
    ap.add_argument("--step",      type=int,   default=10)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--fps",       type=float, default=20.0)
    ap.add_argument("--device",    default="auto",
                    choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    # Tentukan daftar file
    if args.files:
        file_list = []
        for f in args.files:
            p = Path(f)
            if not p.exists():
                print("[WARN] File tidak ditemukan: {}".format(f))
            else:
                file_list.append(str(p))
        if not file_list:
            sys.exit("[ERROR] Tidak ada file valid.")
    else:
        skel_dir = Path(args.skeleton_dir)
        if not skel_dir.exists():
            sys.exit("[ERROR] Direktori tidak ditemukan: {}".format(args.skeleton_dir))
        if skel_dir.is_file() and skel_dir.suffix == ".skeleton":
            # User salah kasih file bukan folder — tangani gracefully
            file_list = [str(skel_dir)]
        else:
            file_list = pick_clips_from_dir(
                str(skel_dir), args.n_clips, args.mix_mode, args.seed)

    if not file_list:
        sys.exit("[ERROR] Tidak ada file skeleton yang dapat dimuat.")

    run_multi_viz(args, file_list)


if __name__ == "__main__":
    main()
