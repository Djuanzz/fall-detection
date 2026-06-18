"""
eval_video_e2e.py
=================
Pengujian Deteksi Jatuh Menggunakan Video — END-TO-END (Subbab 4.1.3).

Berbeda dengan test_fall_yolo17.py yang membaca skeleton .npy yang SUDAH jadi
(klip penuh di-pad/crop 150 frame -> 1x inferensi), skrip ini menguji SISTEM
sebagaimana dipakai saat deployment:

    Video .avi mentah
      -> YOLO11n-pose live (per frame, best-person + EMA smoothing)
      -> sliding window (64 frame, step 15)
      -> BlockGCN per window
      -> agregasi per-video (default: peak / max P(fall))  [TANPA temporal voting]
      -> keputusan FALL / NOT_FALL per video

Daftar video diambil dari val_label.pkl (set uji yang SAMA dengan tabel offline),
lalu dipetakan ke <video_dir>/<nama>.avi.

Skrip ini juga mengukur kecepatan pipeline (rata-rata ms YOLO/frame,
ms BlockGCN/window, ms/frame efektif, FPS efektif) -> untuk tabel kecepatan
sistem end-to-end.

Cara pakai (default = konfigurasi final 04_17ful):
    python scripts/eval_video_e2e.py
    python scripts/eval_video_e2e.py --limit 40          # uji cepat 40 video
    python scripts/eval_video_e2e.py --agg max --threshold 0.5
    python scripts/eval_video_e2e.py \
        --weights weights_new/04_17bal/runs-XX.pt \
        --config  weights_new/04_17bal/config.yaml \
        --label-pkl dataset/yolo17_data/balanced/joint/val_label.pkl \
        --out docs/eval/hasil_e2e_video_04bal.txt
"""

import argparse
import importlib
import pickle
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

# ── Default ──────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS   = "weights_new/04_17ful/runs-38-4142.pt"
DEFAULT_CONFIG    = "weights_new/04_17ful/config.yaml"
DEFAULT_YOLO      = "yolo11n-pose.pt"
DEFAULT_VIDEO_DIR = "dataset/ntu_videos"
DEFAULT_LABEL_PKL = "dataset/yolo17_data/full/joint/val_label.pkl"
DEFAULT_OUT       = "docs/eval/hasil_e2e_video_04.txt"

DEFAULT_WINDOW    = 64
DEFAULT_STEP      = 15
DEFAULT_THRESHOLD = 0.5
DEFAULT_IMGSZ     = 640
DEFAULT_SMOOTH    = 0.4

NUM_JOINTS = 17
LABEL_NAMES = {0: "not_fall", 1: "fall"}


# ── Utilities ────────────────────────────────────────────────────────────────

def import_class(name):
    parts = name.split('.')
    mod = importlib.import_module('.'.join(parts[:-1]))
    return getattr(mod, parts[-1])


def load_model(cfg, weights_path, device):
    Model = import_class(cfg["model"])
    model = Model(**cfg["model_args"])
    weights = torch.load(weights_path, map_location="cpu", weights_only=False)
    weights = {k.replace("module.", ""): v for k, v in weights.items()}
    model.load_state_dict(weights)
    model.to(device).eval()
    return model


def normalize_skeleton_sequence(buffer_np):
    """Translasi hip-center ke origin, scale dari lebar bahu. (T,17,3) -> (T,17,3)."""
    sk = buffer_np.copy()
    xy = sk[:, :, :2]
    hc = (xy[:, 11] + xy[:, 12]) / 2.0
    xy -= hc[:, np.newaxis, :]
    d = np.linalg.norm(xy[:, 5] - xy[:, 6], axis=1)
    sc = d[d > 1e-5].mean() if (d > 1e-5).any() else 1.0
    if sc > 1e-5:
        xy /= sc
    sk[:, :, :2] = xy
    return sk


# ── Ekstraksi keypoint per video (YOLO live) ─────────────────────────────────

def extract_keypoints(yolo, video_path, device, imgsz, smooth_alpha, half):
    """Return (all_kp (T,17,3), n_frames, total_yolo_ms)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0, 0.0

    all_kp = []
    smoothed = None
    yolo_ms = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        kf_xy = np.zeros((NUM_JOINTS, 2), np.float32)
        kf_conf = np.zeros(NUM_JOINTS, np.float32)

        t0 = time.perf_counter()
        results = yolo.predict(frame, verbose=False, device=device,
                               half=half, imgsz=imgsz)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        yolo_ms += (time.perf_counter() - t0) * 1000.0

        if results and results[0].keypoints is not None:
            kps = results[0].keypoints
            if len(kps) > 0 and kps.conf is not None:
                conf_all = kps.conf.cpu().numpy()
                best = int(conf_all.mean(axis=1).argmax())
                raw_xy = kps.xy.cpu().numpy()[best]
                kf_conf = conf_all[best]
                if smoothed is None:
                    smoothed = raw_xy.copy()
                else:
                    hi = kf_conf > 0.3
                    smoothed[hi] = (smooth_alpha * raw_xy[hi] +
                                    (1 - smooth_alpha) * smoothed[hi])
                kf_xy = smoothed.copy()
            else:
                smoothed = None
        else:
            smoothed = None

        kf = np.zeros((NUM_JOINTS, 3), np.float32)
        kf[:, :2] = kf_xy
        kf[:, 2] = kf_conf
        all_kp.append(kf)

    cap.release()
    if not all_kp:
        return None, 0, yolo_ms
    return np.array(all_kp, np.float32), len(all_kp), yolo_ms


# ── Inferensi sliding-window ──────────────────────────────────────────────────

def infer_video(model, all_kp, window_size, step, device):
    """Return (list p_fall per window, total_gcn_ms, n_windows)."""
    T = all_kp.shape[0]
    p_falls = []
    gcn_ms = 0.0

    if T < window_size:
        # video lebih pendek dari window: pakai seluruh frame, resample ke window
        starts = [0]
        get_window = lambda s: all_kp
    else:
        starts = list(range(0, T - window_size + 1, step))
        get_window = lambda s: all_kp[s:s + window_size]

    for s in starts:
        window = get_window(s)
        norm = normalize_skeleton_sequence(window)
        x = norm.astype(np.float32).transpose(2, 0, 1)[:, :, :, np.newaxis]  # (C,T,V,M)

        conf = x[2, :, :, 0]
        valid = max(int((conf > 0).any(axis=1).sum()), 1)
        seg = x[:, :valid, :, :]
        if seg.shape[1] != window_size:
            idx = np.linspace(0, seg.shape[1] - 1, window_size, dtype=int)
            seg = seg[:, idx, :, :]

        x_t = torch.from_numpy(seg[np.newaxis]).float().to(device, non_blocking=True)
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = model(x_t)
            probs = torch.softmax(logits, dim=1)[0]
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        gcn_ms += (time.perf_counter() - t0) * 1000.0
        p_falls.append(probs[1].item())

    return p_falls, gcn_ms, len(starts)


def aggregate(p_falls, agg, threshold):
    """Agregasi P(fall) per-window -> (skor_video, pred_video)."""
    if not p_falls:
        return 0.0, 0
    if agg == "max":
        score = float(np.max(p_falls))
    elif agg == "mean":
        score = float(np.mean(p_falls))
    elif agg == "majority":
        n_fall = sum(1 for p in p_falls if p >= threshold)
        # skor untuk AUC: proporsi window fall
        score = n_fall / len(p_falls)
        return score, (1 if n_fall > len(p_falls) / 2 else 0)
    else:
        raise ValueError(agg)
    return score, (1 if score >= threshold else 0)


# ── Metrik ───────────────────────────────────────────────────────────────────

def compute_metrics(labels, preds, scores):
    TP = sum(l == 1 and p == 1 for l, p in zip(labels, preds))
    TN = sum(l == 0 and p == 0 for l, p in zip(labels, preds))
    FP = sum(l == 0 and p == 1 for l, p in zip(labels, preds))
    FN = sum(l == 1 and p == 0 for l, p in zip(labels, preds))
    n = len(labels)
    acc = (TP + TN) / max(n, 1)
    prec = TP / max(TP + FP, 1)
    sens = TP / max(TP + FN, 1)
    spec = TN / max(TN + FP, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-9)
    bal = (sens + spec) / 2.0
    auc = _auc(labels, scores)
    return dict(TP=TP, TN=TN, FP=FP, FN=FN, n=n, acc=acc, bal_acc=bal,
                prec=prec, sensitivity=sens, specificity=spec, f1=f1, auc=auc)


def _auc(labels, scores):
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos = sum(labels); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = fp = 0; prev = None; pts = [(0.0, 0.0)]
    for score, label in pairs:
        if prev is not None and score != prev:
            pts.append((fp / n_neg, tp / n_pos))
        if label == 1: tp += 1
        else: fp += 1
        prev = score
    pts.append((1.0, 1.0))
    return sum((pts[i][0] - pts[i - 1][0]) * (pts[i - 1][1] + pts[i][1]) / 2
               for i in range(1, len(pts)))


# ── Output ───────────────────────────────────────────────────────────────────

def write_output(out_path, rows, m, speed, args):
    lines = []
    lines.append("# Pengujian Deteksi Jatuh Menggunakan Video — END-TO-END (YOLO + BlockGCN)")
    lines.append("# Tanggal     : {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    lines.append("# Model       : {}".format(args.weights))
    lines.append("# YOLO        : {}  (imgsz={})".format(args.yolo, args.imgsz))
    lines.append("# Window/Step : {} / {}".format(args.window, args.step))
    lines.append("# Threshold   : {:.2f}   Agregasi: {}  (TANPA temporal voting)".format(
        args.threshold, args.agg))
    lines.append("# Set uji     : {}".format(args.label_pkl))
    lines.append("# " + "=" * 78)
    lines.append("# RINGKASAN METRIK (per VIDEO end-to-end)")
    lines.append("# " + "=" * 78)
    lines.append("# Total video       : {}  (not_fall={} fall={})".format(
        m['n'], m['TN'] + m['FP'], m['TP'] + m['FN']))
    lines.append("# Accuracy          : {:.4f}  ({:.2f}%)".format(m['acc'], m['acc'] * 100))
    lines.append("# Balanced Accuracy : {:.4f}  ({:.2f}%)".format(m['bal_acc'], m['bal_acc'] * 100))
    lines.append("# Precision         : {:.4f}  ({:.2f}%)".format(m['prec'], m['prec'] * 100))
    lines.append("# Sensitivity/Recall: {:.4f}  ({:.2f}%)".format(m['sensitivity'], m['sensitivity'] * 100))
    lines.append("# Specificity       : {:.4f}  ({:.2f}%)".format(m['specificity'], m['specificity'] * 100))
    lines.append("# F1-Score          : {:.4f}  ({:.2f}%)".format(m['f1'], m['f1'] * 100))
    lines.append("# AUC-ROC           : {:.4f}".format(m['auc']))
    lines.append("# " + "-" * 50)
    lines.append("# Confusion Matrix:")
    lines.append("#                    Pred NOT_FALL  Pred FALL")
    lines.append("#   True NOT_FALL        {:^5}         {:^5}  (TN / FP)".format(m['TN'], m['FP']))
    lines.append("#   True FALL            {:^5}         {:^5}  (FN / TP)".format(m['FN'], m['TP']))
    lines.append("# " + "=" * 78)
    lines.append("# KECEPATAN PIPELINE END-TO-END  (GPU: {})".format(args.device))
    lines.append("# " + "=" * 78)
    lines.append("# Total frame diproses     : {}".format(speed['n_frames']))
    lines.append("# Total window inferensi   : {}".format(speed['n_windows']))
    lines.append("# YOLO11n-pose / frame     : {:.2f} ms".format(speed['yolo_ms_frame']))
    lines.append("# BlockGCN / window        : {:.2f} ms".format(speed['gcn_ms_window']))
    lines.append("# Pipeline / frame efektif : {:.2f} ms  (YOLO + BlockGCN/step)".format(
        speed['eff_ms_frame']))
    lines.append("# FPS efektif (end-to-end) : {:.1f} FPS".format(speed['eff_fps']))
    lines.append("# " + "=" * 78)
    lines.append("")
    W = 38
    lines.append("# {:<{w}} | {:<10} | {:<10} | {:<7} | skor_fall | n_win".format(
        "nama_video", "label_asli", "prediksi", "hasil", w=W))
    lines.append("# " + "-" * 90)
    for r in rows:
        hasil = "BENAR" if r['label'] == r['pred'] else "SALAH"
        lines.append("{:<{w}} | {:<10} | {:<10} | {:<7} | {:>8.4f} | {:>4d}".format(
            r['name'], LABEL_NAMES[r['label']], LABEL_NAMES[r['pred']],
            hasil, r['score'], r['n_win'], w=W))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Uji deteksi jatuh end-to-end (YOLO + BlockGCN)")
    ap.add_argument("--weights",    default=DEFAULT_WEIGHTS)
    ap.add_argument("--config",     default=DEFAULT_CONFIG)
    ap.add_argument("--yolo",       default=DEFAULT_YOLO)
    ap.add_argument("--video-dir",  default=DEFAULT_VIDEO_DIR)
    ap.add_argument("--label-pkl",  default=DEFAULT_LABEL_PKL)
    ap.add_argument("--out",        default=DEFAULT_OUT)
    ap.add_argument("--window",     type=int,   default=DEFAULT_WINDOW)
    ap.add_argument("--step",       type=int,   default=DEFAULT_STEP)
    ap.add_argument("--threshold",  type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--imgsz",      type=int,   default=DEFAULT_IMGSZ)
    ap.add_argument("--smooth-alpha", type=float, default=DEFAULT_SMOOTH)
    ap.add_argument("--agg",        default="max", choices=["max", "mean", "majority"])
    ap.add_argument("--device",     default="cuda:0")
    ap.add_argument("--no-half",    action="store_true")
    ap.add_argument("--limit",      type=int, default=0, help="Uji N video pertama saja (0=semua)")
    ap.add_argument("--ext",        default=".avi", help="Ekstensi file video")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA tidak tersedia, pakai CPU")
        args.device = "cpu"
    use_cuda = args.device.startswith("cuda")
    half = use_cuda and not args.no_half
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    # ── Daftar video uji dari label pkl ──────────────────────────────────────
    with open(args.label_pkl, "rb") as f:
        names, labels = pickle.load(f)
    video_dir = Path(args.video_dir)
    samples = []
    missing = 0
    for name, label in zip(names, labels):
        vp = video_dir / (name + args.ext)
        if vp.exists():
            samples.append((name, int(label), vp))
        else:
            missing += 1
    if missing:
        print(f"[WARN] {missing} video tidak ditemukan di {video_dir}, dilewati.")
    if args.limit > 0:
        samples = samples[:args.limit]
    n_fall = sum(s[1] for s in samples)
    print(f"\n{'='*64}")
    print("  UJI END-TO-END (YOLO + BlockGCN) — Pengujian Menggunakan Video")
    print(f"{'='*64}")
    print(f"  Video uji  : {len(samples)}  (not_fall={len(samples)-n_fall}, fall={n_fall})")
    print(f"  Model      : {args.weights}")
    print(f"  Agregasi   : {args.agg}  |  Threshold: {args.threshold}")
    print(f"  Window/Step: {args.window}/{args.step}  |  Device: {args.device}\n")

    # ── Muat YOLO + model ────────────────────────────────────────────────────
    from ultralytics import YOLO
    print(f"Memuat YOLO: {args.yolo}")
    yolo = YOLO(args.yolo)
    yolo.to(args.device)
    print(f"Memuat config: {args.config}")
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    print(f"Memuat model: {args.weights}")
    model = load_model(cfg, args.weights, args.device)
    print("Siap.\n")

    # ── Loop video ───────────────────────────────────────────────────────────
    rows = []
    tot_yolo_ms = tot_gcn_ms = 0.0
    tot_frames = tot_windows = 0
    t_start = time.perf_counter()

    for i, (name, label, vp) in enumerate(samples, 1):
        all_kp, n_fr, yolo_ms = extract_keypoints(
            yolo, vp, args.device, args.imgsz, args.smooth_alpha, half)
        if all_kp is None:
            print(f"  [SKIP] gagal baca: {name}")
            continue
        p_falls, gcn_ms, n_win = infer_video(
            model, all_kp, args.window, args.step, args.device)
        score, pred = aggregate(p_falls, args.agg, args.threshold)

        rows.append(dict(name=name, label=label, pred=pred, score=score, n_win=n_win))
        tot_yolo_ms += yolo_ms; tot_gcn_ms += gcn_ms
        tot_frames += n_fr; tot_windows += n_win

        mark = "OK " if label == pred else "XX "
        print(f"\r  [{i}/{len(samples)}] {mark} {name:34s} "
              f"true={LABEL_NAMES[label]:8s} pred={LABEL_NAMES[pred]:8s} "
              f"score={score:.3f}", end="", flush=True)
        if i % 25 == 0:
            print()
    print()

    elapsed = time.perf_counter() - t_start
    labels_v = [r['label'] for r in rows]
    preds_v = [r['pred'] for r in rows]
    scores_v = [r['score'] for r in rows]
    m = compute_metrics(labels_v, preds_v, scores_v)

    yolo_ms_frame = tot_yolo_ms / max(tot_frames, 1)
    gcn_ms_window = tot_gcn_ms / max(tot_windows, 1)
    eff_ms_frame = yolo_ms_frame + gcn_ms_window / max(args.step, 1)
    speed = dict(n_frames=tot_frames, n_windows=tot_windows,
                 yolo_ms_frame=yolo_ms_frame, gcn_ms_window=gcn_ms_window,
                 eff_ms_frame=eff_ms_frame, eff_fps=1000.0 / max(eff_ms_frame, 1e-6))

    write_output(args.out, rows, m, speed, args)

    # ── Cetak ringkasan ──────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  HASIL — Pengujian Menggunakan Video (END-TO-END)")
    print("=" * 64)
    print(f"  Total video       : {m['n']}  (not_fall={m['TN']+m['FP']} fall={m['TP']+m['FN']})")
    print(f"  Accuracy          : {m['acc']*100:.2f}%")
    print(f"  Balanced Accuracy : {m['bal_acc']*100:.2f}%")
    print(f"  Precision         : {m['prec']*100:.2f}%")
    print(f"  Sensitivity/Recall: {m['sensitivity']*100:.2f}%")
    print(f"  Specificity       : {m['specificity']*100:.2f}%")
    print(f"  F1-Score          : {m['f1']*100:.2f}%")
    print(f"  AUC-ROC           : {m['auc']:.4f}")
    print(f"  Confusion: TN={m['TN']} FP={m['FP']} FN={m['FN']} TP={m['TP']}")
    print("-" * 64)
    print("  KECEPATAN PIPELINE END-TO-END:")
    print(f"  YOLO11n-pose / frame     : {yolo_ms_frame:.2f} ms")
    print(f"  BlockGCN / window        : {gcn_ms_window:.2f} ms")
    print(f"  Pipeline / frame efektif : {eff_ms_frame:.2f} ms")
    print(f"  FPS efektif (end-to-end) : {speed['eff_fps']:.1f} FPS")
    print(f"  Total waktu uji          : {elapsed/60:.1f} menit")
    print("=" * 64)
    print(f"\n  Tersimpan di: {args.out}")


if __name__ == "__main__":
    main()