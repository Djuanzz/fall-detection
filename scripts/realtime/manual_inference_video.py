"""
manual_inference_video.py
==========================
Inference satu file video menggunakan YOLO + BlockGCN (YOLO 17-joint pipeline).

Pipeline:
    Video → YOLO11n-pose (per frame) → skeleton buffer →
    sliding window → BlockGCN → tabel prediksi per window + summary

Keputusan akhir = PEAK (FALL bila max P(fall) di seluruh window >= threshold),
konsisten dengan deployment realtime (jatuh = event transien, cukup satu window
yakin "fall"). Majority/rata-rata hanya ditampilkan sebagai pembanding, BUKAN
penentu (rata-rata mengencerkan sinyal jatuh yang sebentar).

Cara pakai:
    python scripts/manual_inference_video.py --video dataset/urfd_videos/fall-05-cam0.mp4
    python scripts/manual_inference_video.py --video dataset/urfd_videos/fall-05-cam0.mp4 --smooth-alpha 1.0    
    python scripts/manual_inference_video.py --video video.mp4 --weights ../weights/xx/runs-*.pt
    python scripts/manual_inference_video.py --video video.mp4 --threshold 0.4 --step 15 --save
"""

import argparse
import importlib
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# DEFAULT_WEIGHTS    = "weights_base/03_17bal/runs-24-3768.pt"
# DEFAULT_CONFIG     = "config/fall-detection-yolo/balanced.yaml"
DEFAULT_WEIGHTS    = "weights_new/04_17ful/runs-38-4142.pt"
DEFAULT_CONFIG     = "weights_new/04_17ful/config.yaml"
DEFAULT_YOLO       = "yolo11n-pose.pt"
DEFAULT_THRESHOLD  = 0.5
DEFAULT_WINDOW     = 64
DEFAULT_STEP       = 15
DEFAULT_YOLO_IMGSZ = 640
DEFAULT_SMOOTH_ALPHA = 0.4

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


def load_model(cfg, weights_path, device, model_override=None):
    model_class = model_override or cfg["model"]
    Model   = import_class(model_class)
    model   = Model(**cfg["model_args"])
    weights = torch.load(weights_path, map_location="cpu", weights_only=False)
    weights = {k.replace("module.", ""): v for k, v in weights.items()}
    model.load_state_dict(weights)
    model.to(device).eval()
    return model


def normalize_skeleton_sequence(buffer_np):
    """Translasi hip center ke origin, scale dari jarak bahu. (T, 17, 3) → (T, 17, 3)."""
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


def run_inference(model, window_size, skeleton_window, device, threshold):
    """Inference satu window (T, 17, 3). Return: (pred, p_fall, p_not_fall)."""
    sk = skeleton_window.astype(np.float32)
    x  = sk.transpose(2, 0, 1)[:, :, :, np.newaxis]  # (C, T, V, M)

    conf  = x[2, :, :, 0]
    valid = int((conf > 0).any(axis=1).sum())
    valid = max(valid, 1)

    seg = x[:, :valid, :, :]
    if seg.shape[1] != window_size:
        idx = np.linspace(0, seg.shape[1] - 1, window_size, dtype=int)
        seg = seg[:, idx, :, :]

    x_t = torch.from_numpy(seg[np.newaxis]).float().to(device, non_blocking=True)
    with torch.inference_mode():
        logits = model(x_t)
        probs  = torch.softmax(logits, dim=1)[0]

    p_nf = probs[0].item()
    p_f  = probs[1].item()
    pred = 1 if p_f >= threshold else 0
    return pred, p_f, p_nf


def draw_skeleton(frame, kf_xy, kf_conf, conf_thresh=0.3):
    h, w = frame.shape[:2]
    for i, j in SKELETON_EDGES:
        if kf_conf[i] > conf_thresh and kf_conf[j] > conf_thresh:
            x1, y1 = int(kf_xy[i][0]), int(kf_xy[i][1])
            x2, y2 = int(kf_xy[j][0]), int(kf_xy[j][1])
            if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                cv2.line(frame, (x1, y1), (x2, y2), COLOR_BLUE, 2)
    for k in range(NUM_JOINTS):
        if kf_conf[k] > conf_thresh:
            x, y = int(kf_xy[k][0]), int(kf_xy[k][1])
            if 0 <= x < w and 0 <= y < h:
                if kf_conf[k] > 0.7:
                    color = COLOR_GREEN
                elif kf_conf[k] > 0.5:
                    color = COLOR_YELLOW
                else:
                    color = COLOR_RED
                cv2.circle(frame, (x, y), 5, color, -1)
                cv2.circle(frame, (x, y), 5, COLOR_WHITE, 1)
    return frame


def annotate_frame(frame, pred, p_fall, threshold, frame_no, total):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 60), (w, h), COLOR_BLACK, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    label = "FALL" if pred == 1 else "NOT FALL"
    color = COLOR_RED if pred == 1 else COLOR_GREEN
    cv2.putText(frame, label, (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

    bar_x, bar_y, bar_w, bar_h = 10, h - 22, w - 20, 14
    fill = int(p_fall * bar_w)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COLOR_GRAY, -1)
    bar_col = COLOR_RED if p_fall >= threshold else COLOR_GREEN
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_col, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COLOR_WHITE, 1)
    thr_x = bar_x + int(threshold * bar_w)
    cv2.line(frame, (thr_x, bar_y - 2), (thr_x, bar_y + bar_h + 2), COLOR_YELLOW, 2)
    cv2.putText(frame, f"P(fall): {p_fall*100:.1f}%  [{frame_no}/{total}]",
                (bar_x, bar_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_WHITE, 1)
    return frame


def main():
    ap = argparse.ArgumentParser(description="Manual inference video — YOLO + BlockGCN")
    ap.add_argument("--video",     required=True, help="Path file video (.mp4/.avi/dll)")
    ap.add_argument("--weights",   default=DEFAULT_WEIGHTS)
    ap.add_argument("--config",    default=DEFAULT_CONFIG)
    ap.add_argument("--yolo",      default=DEFAULT_YOLO)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--window",    type=int,   default=DEFAULT_WINDOW,
                    help="Panjang window frame (default 64)")
    ap.add_argument("--step",      type=int,   default=DEFAULT_STEP,
                    help="Slide window setiap N frame (default 15)")
    ap.add_argument("--device",    default="cuda:0")
    ap.add_argument("--imgsz",     type=int,   default=DEFAULT_YOLO_IMGSZ)
    ap.add_argument("--no-half",   action="store_true")
    ap.add_argument("--save",      action="store_true",
                    help="Simpan video output dengan anotasi skeleton + prediksi")
    ap.add_argument("--out",       default=None,
                    help="Path output video (default: <input>_result.mp4)")
    ap.add_argument("--model",       default=None,
                    help="Override model class, e.g. model.BlockGCN_SE.Model")
    ap.add_argument("--smooth-alpha", type=float, default=DEFAULT_SMOOTH_ALPHA,
                    help="EMA keypoint smoothing 0.1-1.0 (default 0.4)")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA tidak tersedia, pakai CPU")
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
        print(f"ERROR: File tidak ditemukan: {video_path}")
        sys.exit(1)

    # ── Load YOLO ──────────────────────────────────────────────────────────────
    print(f"Memuat YOLO: {args.yolo}")
    try:
        from ultralytics import YOLO
        yolo = YOLO(args.yolo)
        yolo.to(args.device)
    except ImportError:
        print("ERROR: pip install ultralytics")
        sys.exit(1)

    # ── Load config + model ────────────────────────────────────────────────────
    print(f"Memuat config: {args.config}")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    window_size = args.window

    print(f"Memuat model: {args.weights}")
    model = load_model(cfg, args.weights, args.device, model_override=args.model)
    print(f"Model siap di: {args.device}\n")

    # ── Buka video ─────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Tidak bisa membuka video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    actual_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video    : {video_path.name}")
    print(f"Resolusi : {actual_w}x{actual_h}  |  FPS: {video_fps:.1f}  |  Frame: {total_frames}")
    print(f"Window   : {window_size}  |  Step: {args.step}  |  Threshold: {args.threshold}")
    print(f"Device   : {args.device}\n")

    # ── Ekstrak keypoints semua frame ──────────────────────────────────────────
    print("Ekstraksi keypoints dari video...")
    all_keypoints = []
    smoothed_kpts = None

    fn = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fn += 1

        kf_xy   = np.zeros((NUM_JOINTS, 2), np.float32)
        kf_conf = np.zeros(NUM_JOINTS,      np.float32)

        results = yolo.predict(frame, verbose=False, device=args.device,
                               half=yolo_half, imgsz=args.imgsz)
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
                    high_conf = kf_conf > 0.3
                    smoothed_kpts[high_conf] = (
                        args.smooth_alpha * raw_xy[high_conf] +
                        (1 - args.smooth_alpha) * smoothed_kpts[high_conf]
                    )
                kf_xy = smoothed_kpts.copy()
        else:
            smoothed_kpts = None

        kf = np.zeros((NUM_JOINTS, 3), np.float32)
        kf[:, :2] = kf_xy
        kf[:, 2]  = kf_conf
        all_keypoints.append(kf)

        if fn % 50 == 0 or fn == total_frames:
            print(f"  [{fn:4d}/{total_frames}] frame diproses", end="\r")

    cap.release()
    total_frames = len(all_keypoints)
    print(f"\nTotal frame diekstrak: {total_frames}\n")

    if total_frames < window_size:
        print(f"ERROR: Video terlalu pendek ({total_frames} frame < window {window_size})")
        sys.exit(1)

    all_kp = np.array(all_keypoints)  # (T, 17, 3)

    # ── Sliding window inference ───────────────────────────────────────────────
    print("Inference per window...")
    print(f"{'Win':>4}  {'Frame':>12}  {'P(fall)':>8}  Prediksi")
    print("-" * 48)

    window_results = []
    win_idx = 0

    for start in range(0, total_frames - window_size + 1, args.step):
        end    = start + window_size
        window = all_kp[start:end]
        norm   = normalize_skeleton_sequence(window)

        pred, p_fall, p_nf = run_inference(
            model, window_size, norm, args.device, args.threshold)

        win_idx += 1
        label = "FALL" if pred == 1 else "not_fall"
        print(f"{win_idx:>4}  {start:>5}-{end-1:<5}     {p_fall*100:>5.1f}%  {label}")

        window_results.append({"start": start, "end": end - 1,
                                "pred": pred, "p_fall": p_fall, "p_not_fall": p_nf})

    # ── Summary ────────────────────────────────────────────────────────────────
    # Keputusan akhir = PEAK (max P(fall) >= threshold), KONSISTEN dengan
    # deployment realtime: jatuh adalah event transien, jadi cukup SATU window
    # yang yakin "fall" untuk memicu alarm. TIDAK pakai majority/rata-rata yang
    # justru mengencerkan sinyal jatuh yang sebentar.
    preds      = [r["pred"]   for r in window_results]
    p_falls    = [r["p_fall"] for r in window_results]
    n_fall     = sum(preds)
    n_total    = len(preds)
    avg_p_fall = float(np.mean(p_falls))
    max_p_fall = float(np.max(p_falls))
    majority   = 1 if n_fall > n_total / 2 else 0     # ditampilkan hanya sbg pembanding
    final_pred = 1 if max_p_fall >= args.threshold else 0

    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print(f"  Video        : {video_path.name}")
    print(f"  Total frame  : {total_frames}")
    print(f"  Total window : {n_total}")
    print(f"  Window FALL  : {n_fall}/{n_total}  ({n_fall/n_total*100:.1f}%)")
    print(f"  Avg P(fall)  : {avg_p_fall*100:.1f}%   (bukan penentu)")
    print(f"  Max P(fall)  : {max_p_fall*100:.1f}%   <- penentu (peak)")
    print(f"  Majority vote: {'fall' if majority else 'not_fall'}   (pembanding saja)")
    print()
    final_label = "** JATUH **" if final_pred == 1 else "TIDAK JATUH"
    bar = "█" * int(max_p_fall * 40)
    print(f"  >>> PREDIKSI AKHIR (peak, thr={args.threshold}): {final_label}")
    print(f"  Max P(fall): {max_p_fall*100:5.1f}%  {bar}")
    print("=" * 55)

    # ── Simpan video anotasi (opsional) ────────────────────────────────────────
    if args.save:
        out_path = args.out or str(video_path.parent / (video_path.stem + "_result.mp4"))
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_path, fourcc, video_fps, (actual_w, actual_h))
        print(f"\nMenyimpan video anotasi: {out_path}")

        # Peta frame → hasil window terakhir yang mencakup frame itu
        frame_to_result = {}
        for r in window_results:
            for frame_i in range(r["start"], r["end"] + 1):
                frame_to_result[frame_i] = r

        cap2 = cv2.VideoCapture(str(video_path))
        fi   = 0
        last_pred   = 0
        last_p_fall = 0.0
        while True:
            ok, frame = cap2.read()
            if not ok:
                break
            kf = all_kp[fi] if fi < len(all_kp) else np.zeros((NUM_JOINTS, 3), np.float32)
            frame = draw_skeleton(frame, kf[:, :2], kf[:, 2])
            if fi in frame_to_result:
                last_pred   = frame_to_result[fi]["pred"]
                last_p_fall = frame_to_result[fi]["p_fall"]
            frame = annotate_frame(frame, last_pred, last_p_fall,
                                   args.threshold, fi + 1, total_frames)
            writer.write(frame)
            fi += 1
        cap2.release()
        writer.release()
        print(f"Video disimpan: {out_path}")

    print("\nSelesai.")


if __name__ == "__main__":
    main()
