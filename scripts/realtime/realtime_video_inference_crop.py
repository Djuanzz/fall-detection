"""
realtime_video_inference_crop.py
=================================
Varian dari realtime_video_inference.py, tapi tampilan/output DI-CROP hanya ke
orang yang terdeteksi (bukan resolusi penuh 1080/selebar layar). Cocok untuk
bukti deteksi di buku thesis: potongan frame berjalan (sequence/filmstrip) yang
hemat halaman.

Setiap crop diberi overlay ringkas:
    - nomor frame (F123)
    - status FALL / NOT FALL
    - P(fall) dalam persen
    - FPS

Selain live view, script bisa:
    - menyimpan crop teranotasi setiap N frame (--save-every) ke folder sequence
    - merangkai crop tersebut jadi montage grid (--montage) → 1 gambar siap tempel

Cara pakai (dari root repo):
    python scripts/realtime_video_inference_crop.py --video dataset/urfd_videos/merged_output.mp4 --device cuda:0
    python scripts/realtime_video_inference_crop.py --video x.mp4 --save-every 6 --montage --montage-cols 6
    python scripts/realtime_video_inference_crop.py --video x.mp4 --max-speed --no-loop --save-every 5 --montage

Kontrol keyboard:
    q = keluar
    s = simpan crop saat ini
    r = reset buffer + posisi video
    SPACE = pause / resume
"""

import argparse
import collections
import importlib
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# ── Default konfigurasi (samakan dgn realtime_video_inference.py) ──────────────
DEFAULT_WEIGHTS = "weights_new/04_17ful/runs-38-4142.pt"
DEFAULT_CONFIG  = "weights_new/04_17ful/config.yaml"
DEFAULT_MODEL   = None  # None = pakai dari config

DEFAULT_YOLO      = "yolo11n-pose.pt"
DEFAULT_THRESHOLD = 0.5
DEFAULT_WINDOW    = 64
DEFAULT_STEP      = 15
DEFAULT_SMOOTH_ALPHA = 0.4
DEFAULT_YOLO_IMGSZ   = 640

# Crop / sequence
DEFAULT_CROP_PAD    = 0.18   # padding di sekeliling bbox (fraksi dari ukuran bbox)
DEFAULT_CROP_H      = 480    # tinggi kanvas keluaran (px) — TETAP
DEFAULT_CROP_ASPECT = 0.62   # rasio lebar:tinggi kanvas (portrait); lebar = H*aspect
DEFAULT_BOX_SMOOTH  = 0.25   # EMA bbox crop (kecil=lebih stabil, besar=lebih responsif)
DEFAULT_SAVE_EVERY  = 0      # 0 = tidak autosave sequence; >0 = simpan tiap N frame
DEFAULT_MONTAGE_COLS = 6

NUM_JOINTS = 17

SKELETON_EDGES = [
    (0,1),(0,2),(1,3),(2,4),
    (0,5),(0,6),(5,6),
    (5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

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

def load_model(cfg, weights_path, device, model_override=None):
    model_class = model_override if model_override else cfg["model"]
    Model   = import_class(model_class)
    model   = Model(**cfg["model_args"])
    weights = torch.load(weights_path, map_location="cpu", weights_only=False)
    weights = {k.replace("module.", ""): v for k, v in weights.items()}
    model.load_state_dict(weights)
    model.to(device).eval()
    return model


# ── Normalisasi skeleton ───────────────────────────────────────────────────────

def normalize_skeleton_sequence(buffer_np):
    sk = buffer_np.copy()
    xy = sk[:, :, :2]
    hc = (xy[:, 11] + xy[:, 12]) / 2.0
    xy -= hc[:, np.newaxis, :]
    d  = np.linalg.norm(xy[:, 5] - xy[:, 6], axis=1)
    sc = d[d > 1e-5].mean() if (d > 1e-5).any() else 1.0
    if sc > 1e-5:
        xy /= sc
    sk[:, :, :2] = xy
    return sk


# ── Inference BlockGCN ─────────────────────────────────────────────────────────

def run_inference(model, cfg, skeleton_window, device, threshold):
    window_size = cfg["test_feeder_args"]["window_size"]
    sk = skeleton_window.astype(np.float32)
    x  = sk.transpose(2, 0, 1)[:, :, :, np.newaxis]  # (C, T, V, M)

    conf  = x[2, :, :, 0]
    valid = int((conf > 0).any(axis=1).sum())
    valid = max(valid, 1)

    seg = x[:, :valid, :, :]
    if seg.shape[1] != window_size:
        idx = np.linspace(0, seg.shape[1] - 1, window_size, dtype=int)
        seg = seg[:, idx, :, :]

    x = torch.from_numpy(seg[np.newaxis]).float().to(device, non_blocking=True)
    with torch.inference_mode():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0]

    p_nf = probs[0].item()
    p_f  = probs[1].item()
    pred = 1 if p_f >= threshold else 0
    return pred, p_f, p_nf


# ── Visualisasi ────────────────────────────────────────────────────────────────

def draw_skeleton_on_frame(frame, keypoints_xy, keypoints_conf, conf_thresh=0.3):
    h, w = frame.shape[:2]
    for i, j in SKELETON_EDGES:
        if keypoints_conf[i] > conf_thresh and keypoints_conf[j] > conf_thresh:
            x1, y1 = int(keypoints_xy[i][0]), int(keypoints_xy[i][1])
            x2, y2 = int(keypoints_xy[j][0]), int(keypoints_xy[j][1])
            if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                cv2.line(frame, (x1, y1), (x2, y2), COLOR_BLUE, 2)
    for k in range(NUM_JOINTS):
        if keypoints_conf[k] > conf_thresh:
            x, y = int(keypoints_xy[k][0]), int(keypoints_xy[k][1])
            if 0 <= x < w and 0 <= y < h:
                if keypoints_conf[k] > 0.7:
                    color = COLOR_GREEN
                elif keypoints_conf[k] > 0.5:
                    color = COLOR_YELLOW
                else:
                    color = COLOR_RED
                cv2.circle(frame, (x, y), 4, color, -1)
                cv2.circle(frame, (x, y), 4, COLOR_WHITE, 1)
    return frame


def compute_crop_box(bbox, frame_w, frame_h, pad_frac):
    """Bbox (x1,y1,x2,y2) + padding → kotak crop yang sudah di-clamp ke frame."""
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    px = bw * pad_frac
    py = bh * pad_frac
    cx1 = int(max(0, x1 - px))
    cy1 = int(max(0, y1 - py))
    cx2 = int(min(frame_w, x2 + px))
    cy2 = int(min(frame_h, y2 + py))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return cx1, cy1, cx2, cy2


def draw_crop_overlay(crop, pred, p_fall, fps, frame_count, warming, buffer_len,
                      window_size, threshold):
    """Overlay ringkas di crop: nomor frame, status, P(fall), FPS."""
    h, w = crop.shape[:2]

    # bar gelap transparan di atas
    bar_h = 56
    ov = crop.copy()
    cv2.rectangle(ov, (0, 0), (w, bar_h), COLOR_BLACK, -1)
    cv2.addWeighted(ov, 0.55, crop, 0.45, 0, crop)

    # baris 1: nomor frame (kiri)
    cv2.putText(crop, f"F{frame_count}", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)

    if warming:
        cv2.putText(crop, f"WARMUP {buffer_len}/{window_size}", (6, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_YELLOW, 1, cv2.LINE_AA)
        return crop

    # baris 2: status + P(fall)
    if pred == 1:
        status, scolor = "FALL", COLOR_RED
    else:
        status, scolor = "NOT FALL", COLOR_GREEN
    cv2.putText(crop, status, (6, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, scolor, 2, cv2.LINE_AA)

    pf_txt = f"P(fall) {p_fall*100:.1f}%"
    (tw, _), _ = cv2.getTextSize(pf_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    pf_color = COLOR_RED if p_fall >= threshold else COLOR_GREEN
    cv2.putText(crop, pf_txt, (w - tw - 6, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, pf_color, 1, cv2.LINE_AA)

    # bar tipis P(fall) di bawah
    bar_y = h - 8
    fill  = int(p_fall * w)
    cv2.rectangle(crop, (0, bar_y), (w, h), COLOR_GRAY, -1)
    cv2.rectangle(crop, (0, bar_y), (fill, h), pf_color, -1)
    thr_x = int(threshold * w)
    cv2.line(crop, (thr_x, bar_y - 2), (thr_x, h), COLOR_YELLOW, 1)
    return crop


def expand_box_to_aspect(box, aspect, frame_w, frame_h):
    """Lebarkan box (x1,y1,x2,y2) agar rasio w:h == aspect, lalu clamp ke frame.
    Mempertahankan pusat sebisanya supaya orang tetap di tengah."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = x2 - x1
    bh = y2 - y1
    if bw / max(bh, 1e-6) < aspect:
        bw = bh * aspect
    else:
        bh = bw / aspect
    # batasi agar tidak lebih besar dari frame
    bw = min(bw, frame_w)
    bh = min(bh, frame_h)
    # geser pusat supaya box tetap di dalam frame
    cx = min(max(cx, bw / 2.0), frame_w - bw / 2.0)
    cy = min(max(cy, bh / 2.0), frame_h - bh / 2.0)
    nx1 = int(round(cx - bw / 2.0))
    ny1 = int(round(cy - bh / 2.0))
    nx2 = int(round(cx + bw / 2.0))
    ny2 = int(round(cy + bh / 2.0))
    return max(0, nx1), max(0, ny1), min(frame_w, nx2), min(frame_h, ny2)


def letterbox_to_canvas(img, cw, ch, bg=(0, 0, 0)):
    """Resize img ke dalam kanvas (cw, ch) TANPA distorsi (pad bila perlu)."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((ch, cw, 3), bg, np.uint8)
    scale = min(cw / w, ch / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    r = cv2.resize(img, (nw, nh))
    canvas = np.full((ch, cw, 3), bg, np.uint8)
    y0 = (ch - nh) // 2
    x0 = (cw - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    return canvas


def make_montage(frames, cols, cell_pad=6, bg=(255, 255, 255)):
    """Rangkai list crop (tinggi sudah seragam) jadi grid montage."""
    if not frames:
        return None
    cell_h = max(f.shape[0] for f in frames)
    cell_w = max(f.shape[1] for f in frames)
    rows = (len(frames) + cols - 1) // cols
    canvas = np.full(
        ((cell_h + cell_pad) * rows + cell_pad,
         (cell_w + cell_pad) * cols + cell_pad, 3),
        bg, dtype=np.uint8)
    for idx, f in enumerate(frames):
        r, c = divmod(idx, cols)
        y0 = cell_pad + r * (cell_h + cell_pad)
        x0 = cell_pad + c * (cell_w + cell_pad)
        fh, fw = f.shape[:2]
        oy = y0 + (cell_h - fh) // 2
        ox = x0 + (cell_w - fw) // 2
        canvas[oy:oy + fh, ox:ox + fw] = f
    return canvas


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Fall detection per-orang (crop) + sequence/filmstrip")
    ap.add_argument("--video",     required=True)
    ap.add_argument("--weights",   default=DEFAULT_WEIGHTS)
    ap.add_argument("--config",    default=DEFAULT_CONFIG)
    ap.add_argument("--yolo",      default=DEFAULT_YOLO)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--window",    type=int,   default=DEFAULT_WINDOW)
    ap.add_argument("--step",      type=int,   default=DEFAULT_STEP)
    ap.add_argument("--device",    default="cuda:0")
    ap.add_argument("--imgsz",     type=int,   default=DEFAULT_YOLO_IMGSZ)
    ap.add_argument("--no-half",   action="store_true")
    ap.add_argument("--max-speed", action="store_true")
    ap.add_argument("--no-loop",   action="store_true")
    ap.add_argument("--model",     default=DEFAULT_MODEL)
    ap.add_argument("--smooth-alpha", type=float, default=DEFAULT_SMOOTH_ALPHA)
    # crop & sequence
    ap.add_argument("--crop-pad",  type=float, default=DEFAULT_CROP_PAD,
                    help="Padding sekeliling bbox orang (fraksi, default 0.18)")
    ap.add_argument("--crop-h",    type=int,   default=DEFAULT_CROP_H,
                    help="Tinggi kanvas keluaran TETAP (px, default 480)")
    ap.add_argument("--crop-aspect", type=float, default=DEFAULT_CROP_ASPECT,
                    help="Rasio lebar:tinggi kanvas (default 0.62, portrait)")
    ap.add_argument("--box-smooth", type=float, default=DEFAULT_BOX_SMOOTH,
                    help="EMA bbox crop (kecil=stabil, default 0.25)")
    ap.add_argument("--save-every", type=int,  default=DEFAULT_SAVE_EVERY,
                    help="Simpan crop teranotasi tiap N frame (0=off)")
    ap.add_argument("--montage",   action="store_true",
                    help="Rangkai crop tersimpan jadi grid saat selesai")
    ap.add_argument("--montage-cols", type=int, default=DEFAULT_MONTAGE_COLS)
    ap.add_argument("--seqdir",    default="paper_shots/sequence",
                    help="Folder output crop & montage")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA tidak tersedia, pakai CPU (lebih lambat)")
        args.device = "cpu"

    use_cuda  = args.device.startswith("cuda")
    yolo_half = use_cuda and not args.no_half
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: File video tidak ditemukan: {video_path}")
        sys.exit(1)

    print(f"\nMemuat YOLO dari: {args.yolo}")
    try:
        from ultralytics import YOLO
        yolo = YOLO(args.yolo)
        yolo.to(args.device)
    except ImportError:
        print("ERROR: pip install ultralytics")
        sys.exit(1)

    print(f"Memuat config dari: {args.config}")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"Memuat model dari: {args.weights}")
    model = load_model(cfg, args.weights, args.device, model_override=args.model)
    print(f"Model siap di: {args.device}")

    print(f"\nMembuka video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Video tidak bisa dibuka: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS)
    actual_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolusi video : {actual_w}x{actual_h}")
    print(f"FPS video      : {video_fps:.1f}")
    print(f"Total frame    : {total_frames}")
    print(f"Crop tinggi    : {args.crop_h}px, pad {args.crop_pad}")
    print(f"Save sequence  : tiap {args.save_every} frame" if args.save_every
          else "Save sequence  : OFF")
    print("Mulai...\n")

    skeleton_buffer = collections.deque(maxlen=args.window)
    smoothed_kpts   = None

    last_pred       = 0
    last_p_fall     = 0.0

    frame_count    = 0
    save_num       = 0
    paused         = False
    seq_frames     = []     # crop teranotasi untuk montage
    smoothed_box   = None   # EMA bbox crop (float x1,y1,x2,y2)

    # Kanvas keluaran ukuran TETAP → window tidak resize, teks tidak geser
    canvas_h = args.crop_h
    canvas_w = max(1, int(round(args.crop_h * args.crop_aspect)))

    seqdir = ROOT / args.seqdir
    seqdir.mkdir(parents=True, exist_ok=True)

    WINDOW = "Fall Detection (crop) BlockGCN + YOLO11-pose"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, canvas_w, canvas_h)

    fps_times = collections.deque(maxlen=30)
    frame_delay = 1 if args.max_speed else (
        max(1, int(1000 / video_fps)) if video_fps > 0 else 30)

    last_crop = None  # untuk save manual saat pause

    while True:
        if paused:
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = False
            elif key == ord('s') and last_crop is not None:
                p = seqdir / f"crop_{save_num:03d}_f{frame_count}.png"
                cv2.imwrite(str(p), last_crop)
                save_num += 1
                print(f"[SAVE] {p}")
            continue

        ok, frame = cap.read()
        if not ok:
            if args.no_loop:
                print("\nVideo selesai.")
                break
            print("[INFO] Video loop ulang, reset buffer.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            skeleton_buffer.clear()
            smoothed_kpts = None
            smoothed_box = None
            last_pred = 0
            last_p_fall = 0.0
            frame_count = 0
            continue

        frame_count += 1
        t_now = time.time()
        fps_times.append(t_now)
        fps = ((len(fps_times) - 1) / (fps_times[-1] - fps_times[0])
               if len(fps_times) >= 2 else 0.0)

        # ── 1. YOLO ────────────────────────────────────────────────────────────
        kf_xy   = np.zeros((NUM_JOINTS, 2), np.float32)
        kf_conf = np.zeros(NUM_JOINTS,      np.float32)
        bbox    = None

        results = yolo.predict(frame, verbose=False, device=args.device,
                               half=yolo_half, imgsz=args.imgsz)
        person_detected = False
        if results and results[0].keypoints is not None:
            kps = results[0].keypoints
            if len(kps) > 0:
                conf_all = kps.conf.cpu().numpy()
                best     = int(conf_all.mean(axis=1).argmax())
                raw_xy   = kps.xy.cpu().numpy()[best]
                kf_conf  = conf_all[best]
                if (results[0].boxes is not None and
                        len(results[0].boxes) > best):
                    bbox = results[0].boxes.xyxy.cpu().numpy()[best]
                if smoothed_kpts is None:
                    smoothed_kpts = raw_xy.copy()
                else:
                    high = kf_conf > 0.3
                    smoothed_kpts[high] = (
                        args.smooth_alpha * raw_xy[high] +
                        (1 - args.smooth_alpha) * smoothed_kpts[high])
                kf_xy = raw_xy
                person_detected = True
        if not person_detected:
            smoothed_kpts = None

        kf = np.zeros((NUM_JOINTS, 3), np.float32)
        kf[:, :2] = smoothed_kpts if smoothed_kpts is not None else kf_xy
        kf[:, 2]  = kf_conf
        skeleton_buffer.append(kf)

        # ── 2. Gambar skeleton di full frame ───────────────────────────────────
        if person_detected:
            frame = draw_skeleton_on_frame(frame, kf_xy, kf_conf)

        # ── 3. Inference tiap STEP frame ───────────────────────────────────────
        if (len(skeleton_buffer) == args.window and
                frame_count % args.step == 0):
            window_np = np.array(skeleton_buffer)
            valid_frames = int(((window_np[:, :, 2] > 0.3).sum(axis=1) >= 5).sum())
            if valid_frames < args.window // 4:
                last_pred = 0
                last_p_fall = 0.0
            else:
                window_norm = normalize_skeleton_sequence(window_np)
                try:
                    _, p_fall, _ = run_inference(
                        model, cfg, window_norm, args.device, args.threshold)
                    # Label langsung ikut prob: p_fall >= threshold => FALL
                    last_p_fall = p_fall
                    last_pred   = 1 if last_p_fall >= args.threshold else 0
                except Exception as e:
                    print(f"[WARN] Inference error: {e}")

        # ── 4. Crop ke orang + overlay ringkas ─────────────────────────────────
        warming = len(skeleton_buffer) < args.window
        fh, fw = frame.shape[:2]

        raw_box = (compute_crop_box(bbox, fw, fh, args.crop_pad)
                   if bbox is not None else None)

        if raw_box is not None:
            # EMA bbox supaya crop tidak loncat/breathing tiap frame
            rb = np.array(raw_box, np.float32)
            if smoothed_box is None:
                smoothed_box = rb
            else:
                a = args.box_smooth
                smoothed_box = a * rb + (1 - a) * smoothed_box
        # bila tidak ada deteksi: pertahankan box terakhir (jangan loncat ke full)

        if smoothed_box is not None:
            box = expand_box_to_aspect(
                tuple(smoothed_box), args.crop_aspect, fw, fh)
            cx1, cy1, cx2, cy2 = box
            crop = frame[cy1:cy2, cx1:cx2].copy()
        else:
            crop = frame.copy()  # belum pernah ada orang → full frame

        # Kanvas ukuran TETAP (letterbox, tanpa distorsi) → teks stabil
        crop = letterbox_to_canvas(crop, canvas_w, canvas_h)
        crop = draw_crop_overlay(
            crop, last_pred, last_p_fall, fps, frame_count, warming,
            len(skeleton_buffer), args.window, args.threshold)
        last_crop = crop

        # ── 5. Autosave sequence ───────────────────────────────────────────────
        if (args.save_every > 0 and not warming and person_detected and
                frame_count % args.save_every == 0):
            p = seqdir / f"seq_{save_num:03d}_f{frame_count}.png"
            cv2.imwrite(str(p), crop)
            save_num += 1
            if args.montage:
                seq_frames.append(crop.copy())

        # ── 6. Tampilkan ───────────────────────────────────────────────────────
        cv2.imshow(WINDOW, crop)
        key = cv2.waitKey(frame_delay) & 0xFF
        if key == ord('q'):
            print("\nKeluar...")
            break
        elif key == ord(' '):
            paused = True
            print("[PAUSE]")
        elif key == ord('s'):
            p = seqdir / f"crop_{save_num:03d}_f{frame_count}.png"
            cv2.imwrite(str(p), crop)
            save_num += 1
            print(f"[SAVE] {p}")
        elif key == ord('r'):
            skeleton_buffer.clear()
            smoothed_kpts = None
            smoothed_box = None
            last_pred = 0
            last_p_fall = 0.0
            frame_count = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            print("Buffer dan posisi video di-reset.")

    cap.release()
    cv2.destroyAllWindows()

    # ── Montage akhir ──────────────────────────────────────────────────────────
    if args.montage and seq_frames:
        mont = make_montage(seq_frames, args.montage_cols)
        if mont is not None:
            mp = seqdir / "montage.png"
            cv2.imwrite(str(mp), mont)
            print(f"\n[MONTAGE] {len(seq_frames)} crop -> {mp} "
                  f"({args.montage_cols} kolom)")

    print(f"Selesai. Crop tersimpan di: {seqdir}")


if __name__ == "__main__":
    main()
