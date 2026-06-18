"""
realtime_camera_benchmark.py
============================
Real-time fall detection dari kamera (iPhone via Camo / webcam apa pun) yang
SEKALIGUS mengukur & mencetak metrik untuk paper:

    - Kompleksitas model    : params, MACs/FLOPs, ukuran model, peak VRAM
    - Kecepatan / komputasi : latensi YOLO per frame, latensi BlockGCN per window,
                              total pipeline per frame, FPS (proses) & FPS (wall-clock)
    - Statistik sesi        : frame diproses, durasi, jumlah inferensi, deteksi FALL,
                              event jatuh, peak P(fall)

Pipeline (identik dgn realtime_camo_inference.py — TANPA temporal voting):
    Camera -> YOLO11n-pose (per frame) -> EMA smoothing (a=0.4) ->
    sliding-window 64 -> inference tiap STEP=15 frame -> gating ->
    status FALL langsung jika P(fall) >= 0.5

Saat keluar (tekan q), laporan lengkap dicetak ke terminal DAN disimpan ke
docs/eval/hasil_realtime_camera_<timestamp>.txt (siap tempel ke buku TA).

Cara pakai (dari root repo):
    python scripts/realtime_camera_benchmark.py --list-cameras
    python scripts/realtime_camera_benchmark.py
    python scripts/realtime_camera_benchmark.py --camera 1 --device cuda:0
    python scripts/realtime_camera_benchmark.py --mirror

Kontrol keyboard:
    q     = keluar (cetak + simpan laporan)
    s     = simpan screenshot
    r     = reset buffer + reset statistik
    SPACE = pause / resume
"""

import argparse
import collections
import importlib
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# Default konfigurasi FULL (model final tesis)
DEFAULT_WEIGHTS   = "weights_new/04_17ful/runs-38-4142.pt"
DEFAULT_CONFIG    = "weights_new/04_17ful/config.yaml"
DEFAULT_MODEL     = None

DEFAULT_YOLO      = "yolo11n-pose.pt"
DEFAULT_THRESHOLD = 0.5
DEFAULT_WINDOW    = 64
DEFAULT_STEP      = 15
DEFAULT_SMOOTH_ALPHA = 0.4
DEFAULT_MAX_W     = 1280
DEFAULT_MAX_H     = 720
DEFAULT_YOLO_IMGSZ = 640

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


# ----------------------------------------------------------------------------
# Camera helpers
# ----------------------------------------------------------------------------
def list_cameras(max_index: int = 5) -> list:
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
            f"Pastikan Camo Studio / webcam aktif dan terhubung."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def load_model(cfg, weights_path, device, model_override=None):
    model_class = model_override if model_override else cfg["model"]
    Model   = import_class(model_class)
    model   = Model(**cfg["model_args"])
    weights = torch.load(weights_path, map_location="cpu", weights_only=False)
    weights = {k.replace("module.", ""): v for k, v in weights.items()}
    model.load_state_dict(weights)
    model.to(device).eval()
    return model


def measure_complexity(model, cfg, device):
    """Hitung params, MACs/FLOPs, ukuran model (untuk laporan paper)."""
    info = {}
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info["params_total"]     = total
    info["params_trainable"] = trainable
    info["size_fp32_mb"]     = total * 4 / 1e6

    T = cfg["model_args"]["window_size"]
    V = cfg["model_args"]["num_point"]
    x = torch.randn(1, 3, T, V, 1).to(device)

    info["macs"] = None
    info["flops"] = None
    try:
        from thop import profile
        m_cpu = model
        macs, _ = profile(m_cpu, inputs=(x,), verbose=False)
        info["macs"]  = macs
        info["flops"] = macs * 2
    except Exception as e:
        info["macs_error"] = str(e)
    return info


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


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------
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


def draw_status_panel(frame, pred, p_fall, buffer_len, window_size, fps,
                      threshold, frame_count, step_size, paused, cam_index,
                      yolo_ms, gcn_ms, total_ms):
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
            status_text, status_color = "FALL", COLOR_RED
        else:
            status_text, status_color = "NOT FALL", COLOR_GREEN
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
        f"YOLO: {yolo_ms:.1f}ms",
        f"GCN:  {gcn_ms:.1f}ms",
        f"Pipe: {total_ms:.1f}ms",
        f"Thr: {threshold:.2f}",
        f"Step: {step_size}",
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(frame, line, (w - 170, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
    return frame


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def _avg(seq):
    return float(statistics.mean(seq)) if seq else 0.0


def build_report(args, cam_index, actual_w, actual_h, device, complexity,
                 yolo_times, gcn_times, frame_count, elapsed_s,
                 n_infer, n_gated, n_fall_pred, n_fall_events, peak_fall):
    yolo_ms = _avg(yolo_times) * 1000.0
    gcn_ms  = _avg(gcn_times) * 1000.0
    total_ms = yolo_ms + (gcn_ms / max(args.step, 1))
    fps_proc = 1000.0 / total_ms if total_ms > 0 else 0.0
    fps_wall = frame_count / elapsed_s if elapsed_s > 0 else 0.0

    L = []
    a = L.append
    a("=" * 60)
    a("KINERJA PIPELINE REALTIME KAMERA (YOLO11n-pose + BlockGCN)")
    a("Pengukuran end-to-end langsung dari kamera")
    a("=" * 60)
    a(f"Tanggal      : {datetime.now():%Y-%m-%d %H:%M:%S}")
    a(f"Device       : {device}")
    a(f"Model        : {args.weights}")
    a(f"YOLO         : {args.yolo}  (imgsz {args.imgsz})")
    a(f"Kamera       : index {cam_index}  ({actual_w}x{actual_h})")
    a(f"Window/Step  : {args.window} / {args.step}")
    a(f"Threshold    : {args.threshold}")
    a(f"Frame diuji  : {frame_count}  ({elapsed_s:.1f} detik wall-clock)")
    a("")
    a("-" * 60)
    a("KOMPLEKSITAS MODEL (BlockGCN)")
    a("-" * 60)
    a(f"  Total params         : {complexity['params_total']:,} "
      f"({complexity['params_total']/1e6:.3f} M)")
    a(f"  Trainable params     : {complexity['params_trainable']:,}")
    a(f"  Ukuran model (fp32)  : {complexity['size_fp32_mb']:.2f} MB")
    if complexity.get("macs"):
        a(f"  MACs                 : {complexity['macs']/1e9:.3f} G")
        a(f"  FLOPs (2*MACs)       : {complexity['flops']/1e9:.3f} GFLOPs")
    else:
        a(f"  MACs/FLOPs           : (thop tidak tersedia: "
          f"{complexity.get('macs_error','-')})")
    if device.startswith("cuda") and torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated(device) / 1e6
        a(f"  Peak VRAM            : {mem:.1f} MB")
        a(f"  GPU                  : {torch.cuda.get_device_name(0)}")
    a("")
    a("-" * 60)
    a("KECEPATAN / KOMPUTASI")
    a("-" * 60)
    a(f"  YOLO11n-pose per frame     : {yolo_ms:8.2f} ms  "
      f"(n={len(yolo_times)})")
    a(f"  BlockGCN per window        : {gcn_ms:8.2f} ms  "
      f"(n={len(gcn_times)})")
    a(f"  Total pipeline per frame   : {total_ms:8.2f} ms  "
      f"(= YOLO + BlockGCN/step)")
    a(f"  Latency deteksi per frame  : {total_ms:8.2f} ms")
    a(f"  FPS (kemampuan proses)     : {fps_proc:8.1f} FPS")
    a(f"  FPS (wall-clock observasi) : {fps_wall:8.1f} FPS")
    a("")
    a("Tabel siap-tempel:")
    a("  YOLO (ms) | BlockGCN (ms) | Total/frame (ms) | FPS (proses)")
    a(f"  {yolo_ms:.2f}     | {gcn_ms:.2f}          "
      f"| {total_ms:.2f}            | {fps_proc:.1f}")
    a("")
    a("-" * 60)
    a("STATISTIK SESI")
    a("-" * 60)
    a(f"  Total inferensi BlockGCN   : {n_infer}")
    a(f"  Window di-gating (skip)    : {n_gated}")
    a(f"  Window prediksi FALL       : {n_fall_pred}")
    a(f"  Event jatuh terdeteksi     : {n_fall_events} "
      f"(transisi NOT_FALL->FALL)")
    a(f"  Peak P(fall) sesi          : {peak_fall:.4f}")
    a("=" * 60)
    return "\n".join(L)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Real-time fall detection + benchmark untuk paper")
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
    ap.add_argument("--width",     type=int,   default=DEFAULT_MAX_W)
    ap.add_argument("--height",    type=int,   default=DEFAULT_MAX_H)
    ap.add_argument("--model",     default=DEFAULT_MODEL)
    ap.add_argument("--smooth-alpha", type=float, default=DEFAULT_SMOOTH_ALPHA)
    ap.add_argument("--warmup",    type=int, default=300,
                    help="Frame awal yang dibuang dari statistik timing "
                         "(hindari bias inisialisasi CUDA).")
    ap.add_argument("--out", default=None,
                    help="Path file laporan. Default docs/eval/"
                         "hasil_realtime_camera_<timestamp>.txt")
    args = ap.parse_args()

    if args.list_cameras:
        print(f"Available camera indices: {list_cameras()}")
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

    if args.camera is None:
        cams = list_cameras()
        if not cams:
            print("ERROR: Tidak ada kamera terdeteksi. Pastikan Camo/webcam aktif.")
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

    # Kompleksitas model (sekali di awal)
    print("Menghitung kompleksitas model (params/FLOPs)...")
    complexity = measure_complexity(model, cfg, args.device)
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(args.device)

    # Buka kamera
    print(f"\nMembuka kamera index {cam_index}...")
    cap = open_camera(cam_index, args.cam_width, args.cam_height, args.cam_fps)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Resolusi camera: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")
    print(f"\nWindow/Step {args.window}/{args.step} | Thr {args.threshold} | "
          f"Device {args.device} | imgsz {args.imgsz}")
    print(f"Warmup timing : {args.warmup} frame pertama tidak dihitung")
    print("Mulai... (q=quit+laporan, s=screenshot, r=reset, SPACE=pause)\n")

    # State
    skeleton_buffer = collections.deque(maxlen=args.window)
    smoothed_kpts   = None
    last_pred, last_p_fall = 0, 0.0

    frame_count    = 0
    screenshot_num = 0
    paused         = False

    fps_times = collections.deque(maxlen=30)
    last_frame = None

    # Akumulator metrik
    yolo_times, gcn_times = [], []
    last_yolo_ms = last_gcn_ms = last_total_ms = 0.0
    n_infer = n_gated = n_fall_pred = n_fall_events = 0
    peak_fall = 0.0
    prev_pred = 0
    t_start = time.time()

    def reset_stats():
        nonlocal yolo_times, gcn_times, n_infer, n_gated, n_fall_pred
        nonlocal n_fall_events, peak_fall, prev_pred, frame_count, t_start
        yolo_times, gcn_times = [], []
        n_infer = n_gated = n_fall_pred = n_fall_events = 0
        peak_fall = 0.0
        prev_pred = 0
        frame_count = 0
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(args.device)
        t_start = time.time()

    try:
        while True:
            if paused:
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = False
                elif key == ord('s') and last_frame is not None:
                    ss = f"camera_bench_{screenshot_num:03d}.jpg"
                    cv2.imwrite(ss, last_frame)
                    screenshot_num += 1
                    print(f"Screenshot disimpan: {ss}")
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
            fps = ((len(fps_times) - 1) / (fps_times[-1] - fps_times[0])
                   if len(fps_times) >= 2 else 0.0)

            measure = frame_count > args.warmup

            # ---- YOLO pose (timed) ----
            kf_xy   = np.zeros((NUM_JOINTS, 2), np.float32)
            kf_conf = np.zeros(NUM_JOINTS,      np.float32)

            if use_cuda:
                torch.cuda.synchronize()
            t_y0 = time.time()
            results = yolo.predict(frame, verbose=False, device=args.device,
                                   half=yolo_half, imgsz=args.imgsz)
            if use_cuda:
                torch.cuda.synchronize()
            last_yolo_ms = (time.time() - t_y0) * 1000.0
            if measure:
                yolo_times.append(last_yolo_ms / 1000.0)

            person_detected = False
            if results and results[0].keypoints is not None:
                kps = results[0].keypoints
                if len(kps) > 0:
                    conf_all = kps.conf.cpu().numpy()
                    best     = int(conf_all.mean(axis=1).argmax())
                    raw_xy   = kps.xy.cpu().numpy()[best]
                    kf_conf  = conf_all[best]
                    if smoothed_kpts is None:
                        smoothed_kpts = raw_xy.copy()
                    else:
                        hc = kf_conf > 0.3
                        smoothed_kpts[hc] = (
                            args.smooth_alpha * raw_xy[hc] +
                            (1 - args.smooth_alpha) * smoothed_kpts[hc])
                    kf_xy = raw_xy
                    person_detected = True
            if not person_detected:
                smoothed_kpts = None

            kf = np.zeros((NUM_JOINTS, 3), np.float32)
            kf[:, :2] = smoothed_kpts if smoothed_kpts is not None else kf_xy
            kf[:, 2]  = kf_conf
            skeleton_buffer.append(kf)

            if person_detected:
                frame = draw_skeleton_on_frame(frame, kf_xy, kf_conf)

            # ---- BlockGCN inference (timed) tiap STEP frame ----
            if (len(skeleton_buffer) == args.window and
                    frame_count % args.step == 0):
                window_np = np.array(skeleton_buffer)
                valid_frames = int(((window_np[:, :, 2] > 0.3).sum(axis=1) >= 5).sum())
                if valid_frames < args.window // 4:
                    last_pred, last_p_fall = 0, 0.0
                    if measure:
                        n_gated += 1
                else:
                    window_norm = normalize_skeleton_sequence(window_np)
                    try:
                        if use_cuda:
                            torch.cuda.synchronize()
                        t_g0 = time.time()
                        _, p_fall, _ = run_inference(
                            model, cfg, window_norm, args.device, args.threshold)
                        if use_cuda:
                            torch.cuda.synchronize()
                        last_gcn_ms = (time.time() - t_g0) * 1000.0
                        if measure:
                            gcn_times.append(last_gcn_ms / 1000.0)
                            n_infer += 1

                        last_pred   = 1 if p_fall >= args.threshold else 0
                        last_p_fall = p_fall
                        if measure:
                            peak_fall = max(peak_fall, p_fall)
                            if last_pred == 1:
                                n_fall_pred += 1
                            if last_pred == 1 and prev_pred == 0:
                                n_fall_events += 1
                            prev_pred = last_pred
                    except Exception as e:
                        print(f"[WARN] Inference error: {e}")

            last_total_ms = last_yolo_ms + (last_gcn_ms / max(args.step, 1))

            frame = draw_status_panel(
                frame, pred=last_pred, p_fall=last_p_fall,
                buffer_len=len(skeleton_buffer), window_size=args.window,
                fps=fps, threshold=args.threshold, frame_count=frame_count,
                step_size=args.step, paused=paused, cam_index=cam_index,
                yolo_ms=last_yolo_ms, gcn_ms=last_gcn_ms, total_ms=last_total_ms)

            h, w = frame.shape[:2]
            scale = min(args.width / w, args.height / h)
            frame_resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
            last_frame = frame_resized
            cv2.imshow("Fall Detection - Camera Benchmark", frame_resized)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nKeluar...")
                break
            elif key == ord(' '):
                paused = True
                print("[PAUSE]")
            elif key == ord('s'):
                ss = f"camera_bench_{screenshot_num:03d}.jpg"
                cv2.imwrite(ss, frame_resized)
                screenshot_num += 1
                print(f"Screenshot disimpan: {ss}")
            elif key == ord('r'):
                skeleton_buffer.clear()
                smoothed_kpts = None
                last_pred, last_p_fall = 0, 0.0
                reset_stats()
                print("Buffer + statistik di-reset.")
    finally:
        elapsed_s = time.time() - t_start
        cap.release()
        cv2.destroyAllWindows()

        report = build_report(
            args, cam_index, actual_w, actual_h, args.device, complexity,
            yolo_times, gcn_times, frame_count, elapsed_s,
            n_infer, n_gated, n_fall_pred, n_fall_events, peak_fall)
        print("\n" + report)

        if args.out:
            out_path = Path(args.out)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = ROOT / f"docs/eval/hasil_realtime_camera_{ts}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"\n[Laporan tersimpan ke: {out_path}]")


if __name__ == "__main__":
    main()
