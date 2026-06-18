"""
realtime_camo_inference.py
==========================
Real-time fall detection menggunakan iPhone via Camo virtual webcam.

Pipeline:
    Camo (iPhone) → OpenCV VideoCapture → YOLO11n-pose (per frame) →
    buffer skeleton → setiap STEP_SIZE frame → BlockGCN → tampilkan prediksi

Cara pakai (dari folder scripts/):
    # List kamera dulu untuk tahu index Camo
    python scripts/realtime_camo_inference.py --list-cameras

    # Auto-pick (default: index terakhir, biasanya Camo)
    python scripts/realtime_camo_inference.py

    # Pilih index manual
    python scripts/realtime_camo_inference.py --camera 1
    python scripts/realtime_camo_inference.py --camera 1 --device cuda:0

Kontrol keyboard:
    q     = keluar
    s     = simpan screenshot
    r     = reset buffer
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

# Path setup
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# Default konfigurasi FULL
DEFAULT_WEIGHTS   = "weights_new/04_17ful/runs-38-4142.pt"
DEFAULT_CONFIG    = "weights_new/04_17ful/config.yaml"

# Default konfigurasi BALANCED (uncomment kalau mau pakai)
# DEFAULT_WEIGHTS   = "weights_base/03_17bal/runs-24-3768.pt"
# DEFAULT_CONFIG    = "config/fall-detection-yolo/balanced.yaml"
DEFAULT_MODEL     = None

DEFAULT_YOLO      = "yolo11n-pose.pt"
DEFAULT_THRESHOLD = 0.5
DEFAULT_WINDOW    = 64
DEFAULT_STEP      = 15
DEFAULT_SMOOTH_ALPHA = 0.4
DEFAULT_MAX_W     = 1280
DEFAULT_MAX_H     = 720
DEFAULT_YOLO_IMGSZ = 640

# Camo defaults
DEFAULT_CAM_W   = 1280
DEFAULT_CAM_H   = 720
DEFAULT_CAM_FPS = 30

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


# Camera helpers
def list_cameras(max_index: int = 5) -> list[int]:
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                available.append(i)
            cap.release()
    return available


def open_camera(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera index {index}. "
            f"Pastikan Camo Studio jalan dan iPhone terhubung."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def load_model(cfg, weights_path, device, model_override=None):
    model_class = model_override if model_override else cfg["model"]
    Model   = import_class(model_class)
    model   = Model(**cfg["model_args"])
    weights = torch.load(weights_path, map_location="cpu", weights_only=False)
    weights = {k.replace("module.", ""): v for k, v in weights.items()}
    model.load_state_dict(weights)
    model.to(device).eval()
    return model



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


def run_inference(model, cfg, skeleton_window, device, threshold):
    window_size = cfg["test_feeder_args"]["window_size"]

    sk = skeleton_window.astype(np.float32)
    x  = sk.transpose(2, 0, 1)[:, :, :, np.newaxis]

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
                cv2.circle(frame, (x, y), 5, color, -1)
                cv2.circle(frame, (x, y), 5, COLOR_WHITE, 1)
    return frame


def draw_status_panel(frame, pred, p_fall, p_not_fall,
                      buffer_len, window_size, fps, threshold,
                      frame_count, step_size, paused, cam_index):
    h, w = frame.shape[:2]
    panel_h = 100

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - panel_h), (w, h), COLOR_BLACK, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    is_warming = buffer_len < window_size

    if is_warming:
        progress = int((buffer_len / window_size) * (w - 20))
        cv2.rectangle(frame, (10, h - panel_h + 10),
                      (10 + progress, h - panel_h + 30), COLOR_GRAY, -1)
        cv2.rectangle(frame, (10, h - panel_h + 10),
                      (w - 10, h - panel_h + 30), COLOR_WHITE, 1)
        text = f"WARMING UP... {buffer_len}/{window_size} frame"
        cv2.putText(frame, text, (15, h - panel_h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1)
    else:
        if pred == 1:
            status_text  = "FALL"
            status_color = COLOR_RED
        else:
            status_text  = "NOT FALL"
            status_color = COLOR_GREEN

        cv2.putText(frame, status_text, (10, h - panel_h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2)

        bar_x, bar_y   = 10, h - panel_h + 40
        bar_w, bar_h_b = w - 20, 16
        fill_w = int(p_fall * bar_w)

        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h_b), COLOR_GRAY, -1)
        bar_color = COLOR_RED if p_fall >= threshold else COLOR_GREEN
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + fill_w, bar_y + bar_h_b), bar_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h_b), COLOR_WHITE, 1)

        thr_x = bar_x + int(threshold * bar_w)
        cv2.line(frame, (thr_x, bar_y - 3), (thr_x, bar_y + bar_h_b + 3),
                 COLOR_YELLOW, 2)

        cv2.putText(frame, f"P(fall): {p_fall*100:.1f}%",
                    (bar_x, bar_y + bar_h_b + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WHITE, 1)

    info_lines = [
        f"FPS: {fps:.1f}",
        f"{'[PAUSE]' if paused else f'frame {frame_count}'}",
        f"Cam: {cam_index}",
        f"Thr: {threshold:.2f}",
        f"Step: {step_size}",
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(frame, line, (w - 160, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
    return frame


def main():
    ap = argparse.ArgumentParser(
        description="Real-time fall detection dari iPhone via Camo virtual webcam")
    ap.add_argument("--camera",    type=int, default=None,
                    help="Index kamera. None = auto-pick (index terakhir).")
    ap.add_argument("--list-cameras", action="store_true",
                    help="List index kamera yang tersedia dan keluar.")
    ap.add_argument("--cam-width",  type=int, default=DEFAULT_CAM_W)
    ap.add_argument("--cam-height", type=int, default=DEFAULT_CAM_H)
    ap.add_argument("--cam-fps",    type=int, default=DEFAULT_CAM_FPS)
    ap.add_argument("--mirror",     action="store_true",
                    help="Flip horizontal (selfie view).")

    ap.add_argument("--weights",   default=DEFAULT_WEIGHTS)
    ap.add_argument("--config",    default=DEFAULT_CONFIG)
    ap.add_argument("--yolo",      default=DEFAULT_YOLO)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--window",    type=int,   default=DEFAULT_WINDOW)
    ap.add_argument("--step",      type=int,   default=DEFAULT_STEP)
    ap.add_argument("--device",    default="cuda:0")
    ap.add_argument("--imgsz",     type=int,   default=DEFAULT_YOLO_IMGSZ)
    ap.add_argument("--no-half",   action="store_true")
    ap.add_argument("--width",     type=int,   default=DEFAULT_MAX_W,
                    help="Lebar window display")
    ap.add_argument("--height",    type=int,   default=DEFAULT_MAX_H,
                    help="Tinggi window display")
    ap.add_argument("--model",       default=DEFAULT_MODEL)
    ap.add_argument("--smooth-alpha", type=float, default=DEFAULT_SMOOTH_ALPHA,
                    help="EMA keypoint smoothing 0.1-1.0 (default 0.4)")
    args = ap.parse_args()

    if args.list_cameras:
        cams = list_cameras()
        print(f"Available camera indices: {cams}")
        return

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA tidak tersedia, pakai CPU (lebih lambat)")
        args.device = "cpu"

    use_cuda = args.device.startswith("cuda")
    yolo_half = use_cuda and not args.no_half
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Pilih camera index
    if args.camera is None:
        cams = list_cameras()
        if not cams:
            print("ERROR: Tidak ada kamera terdeteksi. Pastikan Camo Studio jalan.")
            sys.exit(1)
        cam_index = cams[-1]
        print(f"Auto-pick camera index {cam_index} (tersedia: {cams})")
    else:
        cam_index = args.camera

    # Load YOLO
    print(f"\nMemuat YOLO dari: {args.yolo}")
    try:
        from ultralytics import YOLO
        yolo = YOLO(args.yolo)
        yolo.to(args.device)
    except ImportError:
        print("ERROR: pip install ultralytics")
        sys.exit(1)

    # Load config + model
    print(f"Memuat config dari: {args.config}")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"Memuat model dari: {args.weights}")
    model = load_model(cfg, args.weights, args.device, model_override=args.model)
    print(f"Model siap di: {args.device}")

    # Buka Camo camera
    print(f"\nMembuka Camo camera index {cam_index}...")
    cap = open_camera(cam_index, args.cam_width, args.cam_height, args.cam_fps)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Resolusi camera: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")

    print(f"\nKonfigurasi:")
    print(f"  Window size : {args.window} frame")
    print(f"  Step size   : {args.step} frame")
    print(f"  Threshold   : {args.threshold}")
    print(f"  Device      : {args.device}")
    print(f"  YOLO imgsz  : {args.imgsz}")
    print(f"  YOLO FP16   : {'Ya' if yolo_half else 'Tidak'}")
    print(f"  Mirror      : {'Ya' if args.mirror else 'Tidak'}")
    print("Mulai... (q=quit, s=screenshot, r=reset, SPACE=pause)\n")

    # State
    skeleton_buffer = collections.deque(maxlen=args.window)
    smoothed_kpts   = None

    last_pred       = 0
    last_p_fall     = 0.0
    last_p_not_fall = 1.0

    frame_count    = 0
    screenshot_num = 0
    paused         = False

    fps_times = collections.deque(maxlen=30)
    last_frame = None

    try:
        while True:
            if paused:
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = False
                elif key == ord('s') and last_frame is not None:
                    ss_path = f"camo_screenshot_{screenshot_num:03d}.jpg"
                    cv2.imwrite(ss_path, last_frame)
                    screenshot_num += 1
                    print(f"Screenshot disimpan: {ss_path}")
                continue

            ok, frame = cap.read()
            if not ok:
                print("[WARN] Frame read failed, retry...")
                time.sleep(0.05)
                continue

            if args.mirror:
                frame = cv2.flip(frame, 1)

            frame_count += 1
            t_now = time.time()
            fps_times.append(t_now)

            if len(fps_times) >= 2:
                fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0])
            else:
                fps = 0.0

            # YOLO pose
            kf_xy   = np.zeros((NUM_JOINTS, 2), np.float32)
            kf_conf = np.zeros(NUM_JOINTS,      np.float32)

            results = yolo.predict(
                frame,
                verbose=False,
                device=args.device,
                half=yolo_half,
                imgsz=args.imgsz,
            )
            person_detected = False

            if results and results[0].keypoints is not None:
                kps = results[0].keypoints
                if len(kps) > 0:
                    conf_all = kps.conf.cpu().numpy()
                    best     = int(conf_all.mean(axis=1).argmax())
                    raw_xy   = kps.xy.cpu().numpy()[best]
                    kf_conf  = conf_all[best]
                    # EMA smoothing untuk buffer inference
                    if smoothed_kpts is None:
                        smoothed_kpts = raw_xy.copy()
                    else:
                        high_conf = kf_conf > 0.3
                        smoothed_kpts[high_conf] = (
                            args.smooth_alpha * raw_xy[high_conf] +
                            (1 - args.smooth_alpha) * smoothed_kpts[high_conf]
                        )
                    kf_xy           = raw_xy   # raw untuk display
                    person_detected = True
            if not person_detected:
                smoothed_kpts = None

            kf = np.zeros((NUM_JOINTS, 3), np.float32)
            kf[:, :2] = smoothed_kpts if smoothed_kpts is not None else kf_xy
            kf[:, 2]  = kf_conf
            skeleton_buffer.append(kf)

            if person_detected:
                frame = draw_skeleton_on_frame(frame, kf_xy, kf_conf)

            # Inference setiap STEP_SIZE frame
            if (len(skeleton_buffer) == args.window and
                    frame_count % args.step == 0):
                window_np   = np.array(skeleton_buffer)

                # Gate: skip jika tidak ada orang terdeteksi di window
                valid_frames = int(((window_np[:, :, 2] > 0.3).sum(axis=1) >= 5).sum())
                if valid_frames < args.window // 4:
                    last_pred       = 0
                    last_p_fall     = 0.0
                    last_p_not_fall = 1.0
                else:
                    window_norm = normalize_skeleton_sequence(window_np)

                    try:
                        _, p_fall, _ = run_inference(
                            model, cfg, window_norm,
                            args.device, args.threshold)
                        # Label langsung ikut prob: p_fall >= threshold => FALL
                        last_pred       = 1 if p_fall >= args.threshold else 0
                        last_p_fall     = p_fall
                        last_p_not_fall = 1.0 - p_fall
                    except Exception as e:
                        print(f"[WARN] Inference error: {e}")

            frame = draw_status_panel(
                frame,
                pred         = last_pred,
                p_fall       = last_p_fall,
                p_not_fall   = last_p_not_fall,
                buffer_len   = len(skeleton_buffer),
                window_size  = args.window,
                fps          = fps,
                threshold    = args.threshold,
                frame_count  = frame_count,
                step_size    = args.step,
                paused       = paused,
                cam_index    = cam_index,
            )

            h, w = frame.shape[:2]
            scale = min(args.width / w, args.height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame_resized = cv2.resize(frame, (new_w, new_h))
            last_frame = frame_resized

            cv2.imshow("Fall Detection - Camo iPhone Camera", frame_resized)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nKeluar...")
                break
            elif key == ord(' '):
                paused = True
                print("[PAUSE]")
            elif key == ord('s'):
                ss_path = f"camo_screenshot_{screenshot_num:03d}.jpg"
                cv2.imwrite(ss_path, frame_resized)
                screenshot_num += 1
                print(f"Screenshot disimpan: {ss_path}")
            elif key == ord('r'):
                skeleton_buffer.clear()
                smoothed_kpts = None
                last_pred       = 0
                last_p_fall     = 0.0
                last_p_not_fall = 1.0
                frame_count     = 0
                print("Buffer di-reset.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Selesai.")


if __name__ == "__main__":
    main()
