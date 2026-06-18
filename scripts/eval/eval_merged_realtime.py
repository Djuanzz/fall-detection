"""
eval_merged_realtime.py
=======================
Pengukuran kinerja pipeline near-realtime pada VIDEO GABUNGAN val
(dataset/merged_video/merged_val.mp4) + ground-truth per-frame dari
merged_val_timeline.csv (hasil scripts/merge_val_videos.py).

Menjalankan pipeline SAMA PERSIS dengan realtime_video_inference.py:
    Video -> YOLO11n-pose (per frame) -> buffer 64 frame ->
    tiap STEP frame -> BlockGCN -> status FALL jika P(fall)>=threshold
    (TANPA temporal voting)

FOKUS DEFAULT = KECEPATAN/KOMPUTASI (bukan akurasi). Yang diukur & disimpan:
    YOLO ms/frame, BlockGCN ms/window, total ms/frame, latency, FPS.
Akurasi (tidak butuh di sini, sudah ditangani eval_video_e2e.py) hanya dihitung
bila pakai flag --accuracy + ada file timeline.

Sambil berjalan tampil di layar: skeleton, bar P(fall), status FALL/NOT_FALL,
FPS, dan (bila --accuracy) label ground-truth frame saat ini.

Hasil disimpan ke docs/eval/ (default hasil_merged_val.txt).
Kontrol: q=keluar (hasil parsial tetap disimpan), s=screenshot, SPACE=pause.

Cara pakai (kamu yang run):
    python scripts/eval_merged_realtime.py                     # tampil layar + ukur kecepatan
    python scripts/eval_merged_realtime.py --max-speed         # ukur cepat (proses secepatnya)
    python scripts/eval_merged_realtime.py --no-display --max-speed   # headless, paling cepat
    python scripts/eval_merged_realtime.py --accuracy          # + akurasi (butuh timeline.csv)
"""

import argparse
import collections
import csv
import importlib
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

DEFAULT_VIDEO    = "dataset/merged_video/merged_val_5x.mp4"
DEFAULT_TIMELINE = None  # default: <video_dir>/<stem>_timeline.csv
DEFAULT_WEIGHTS  = "weights_new/04_17ful/runs-38-4142.pt"
DEFAULT_CONFIG   = "weights_new/04_17ful/config.yaml"
DEFAULT_YOLO     = "yolo11n-pose.pt"
DEFAULT_OUT      = "docs/eval/hasil_merged_val.txt"

DEFAULT_THRESHOLD = 0.5
DEFAULT_WINDOW    = 64
DEFAULT_STEP      = 15
DEFAULT_SMOOTH    = 0.4
DEFAULT_IMGSZ     = 640
DEFAULT_SEAM_GUARD = 16   # frame di sekitar sambungan klip yg diabaikan (versi bersih)

NUM_JOINTS = 17
LABEL_NAMES = {0: "not_fall", 1: "fall"}

SKELETON_EDGES = [
    (0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,6),
    (5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]
COLOR_GREEN=(0,220,0); COLOR_RED=(0,0,220); COLOR_YELLOW=(0,200,200)
COLOR_WHITE=(255,255,255); COLOR_BLACK=(0,0,0); COLOR_GRAY=(128,128,128)
COLOR_BLUE=(220,100,0)


# ── Model ────────────────────────────────────────────────────────────────────

def import_class(name):
    parts = name.split('.')
    return getattr(importlib.import_module('.'.join(parts[:-1])), parts[-1])


def load_model(cfg, weights_path, device):
    Model = import_class(cfg["model"])
    model = Model(**cfg["model_args"])
    w = torch.load(weights_path, map_location="cpu", weights_only=False)
    w = {k.replace("module.", ""): v for k, v in w.items()}
    model.load_state_dict(w)
    return model.to(device).eval()


def normalize_skeleton_sequence(buf):
    sk = buf.copy(); xy = sk[:, :, :2]
    hc = (xy[:, 11] + xy[:, 12]) / 2.0
    xy -= hc[:, np.newaxis, :]
    d = np.linalg.norm(xy[:, 5] - xy[:, 6], axis=1)
    sc = d[d > 1e-5].mean() if (d > 1e-5).any() else 1.0
    if sc > 1e-5:
        xy /= sc
    sk[:, :, :2] = xy
    return sk


def run_inference(model, window_size, sk_window, device):
    sk = sk_window.astype(np.float32)
    x = sk.transpose(2, 0, 1)[:, :, :, np.newaxis]
    conf = x[2, :, :, 0]
    valid = max(int((conf > 0).any(axis=1).sum()), 1)
    seg = x[:, :valid, :, :]
    if seg.shape[1] != window_size:
        idx = np.linspace(0, seg.shape[1] - 1, window_size, dtype=int)
        seg = seg[:, idx, :, :]
    x_t = torch.from_numpy(seg[np.newaxis]).float().to(device, non_blocking=True)
    with torch.inference_mode():
        probs = torch.softmax(model(x_t), dim=1)[0]
    return probs[1].item()


# ── Timeline ground-truth ─────────────────────────────────────────────────────

def load_timeline(path):
    """Return (gt_per_frame list, clips list[dict], n_frames)."""
    clips = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            clips.append(dict(name=r["name"], label=int(r["label"]),
                              start=int(r["start_frame"]), end=int(r["end_frame"])))
    n = max((c["end"] for c in clips), default=-1) + 1
    gt = np.full(n, -1, np.int8)          # -1 = tak terdefinisi
    seam = np.zeros(n, bool)
    for c in clips:
        gt[c["start"]:c["end"] + 1] = c["label"]
    for c in clips[1:]:                   # tandai guard di awal tiap klip (sambungan)
        s = c["start"]
        seam[max(0, s - DEFAULT_SEAM_GUARD): s + DEFAULT_SEAM_GUARD] = True
    return gt, seam, clips, n


# ── Metrik ───────────────────────────────────────────────────────────────────

def metrics_from_cm(TP, TN, FP, FN):
    n = TP + TN + FP + FN
    acc = (TP + TN) / max(n, 1)
    prec = TP / max(TP + FP, 1)
    sens = TP / max(TP + FN, 1)
    spec = TN / max(TN + FP, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-9)
    bal = (sens + spec) / 2.0
    return dict(n=n, TP=TP, TN=TN, FP=FP, FN=FN, acc=acc, bal_acc=bal,
                prec=prec, sens=sens, spec=spec, f1=f1)


def confusion(gt, pred, mask):
    TP = int(np.sum((gt == 1) & (pred == 1) & mask))
    TN = int(np.sum((gt == 0) & (pred == 0) & mask))
    FP = int(np.sum((gt == 0) & (pred == 1) & mask))
    FN = int(np.sum((gt == 1) & (pred == 0) & mask))
    return metrics_from_cm(TP, TN, FP, FN)


# ── Visualisasi ───────────────────────────────────────────────────────────────

def draw_skeleton(frame, xy, conf, th=0.3):
    h, w = frame.shape[:2]
    for i, j in SKELETON_EDGES:
        if conf[i] > th and conf[j] > th:
            x1, y1, x2, y2 = int(xy[i][0]), int(xy[i][1]), int(xy[j][0]), int(xy[j][1])
            if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                cv2.line(frame, (x1, y1), (x2, y2), COLOR_BLUE, 2)
    for k in range(NUM_JOINTS):
        if conf[k] > th:
            x, y = int(xy[k][0]), int(xy[k][1])
            if 0 <= x < w and 0 <= y < h:
                c = COLOR_GREEN if conf[k] > 0.7 else COLOR_YELLOW if conf[k] > 0.5 else COLOR_RED
                cv2.circle(frame, (x, y), 5, c, -1)
                cv2.circle(frame, (x, y), 5, COLOR_WHITE, 1)
    return frame


def draw_panel(frame, pred, p_fall, warming, buf_len, win, fps, thr,
               fc, total, step, gt_label, gt_name):
    h, w = frame.shape[:2]
    ph = 100
    ov = frame.copy()
    cv2.rectangle(ov, (0, h - ph), (w, h), COLOR_BLACK, -1)
    cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)

    if warming:
        prog = int((buf_len / win) * (w - 20))
        cv2.rectangle(frame, (10, h - ph + 10), (10 + prog, h - ph + 30), COLOR_GRAY, -1)
        cv2.rectangle(frame, (10, h - ph + 10), (w - 10, h - ph + 30), COLOR_WHITE, 1)
        cv2.putText(frame, f"WARMING UP... {buf_len}/{win}", (15, h - ph + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1)
    else:
        st = "FALL" if pred == 1 else "NOT FALL"
        sc = COLOR_RED if pred == 1 else COLOR_GREEN
        cv2.putText(frame, st, (10, h - ph + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, sc, 2)
        bx, by, bw, bh = 10, h - ph + 40, w - 20, 16
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), COLOR_GRAY, -1)
        bc = COLOR_RED if p_fall >= thr else COLOR_GREEN
        cv2.rectangle(frame, (bx, by), (bx + int(p_fall * bw), by + bh), bc, -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), COLOR_WHITE, 1)
        tx = bx + int(thr * bw)
        cv2.line(frame, (tx, by - 3), (tx, by + bh + 3), COLOR_YELLOW, 2)
        cv2.putText(frame, f"P(fall): {p_fall*100:.1f}%", (bx, by + bh + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WHITE, 1)

    if total > 0:
        py = h - ph - 6
        cv2.rectangle(frame, (0, py), (w, py + 4), COLOR_GRAY, -1)
        cv2.rectangle(frame, (0, py), (int(fc / total * w), py + 4), COLOR_BLUE, -1)

    # Info kiri-bawah
    lines = [f"FPS: {fps:.1f}", f"{fc}/{total}", f"Thr: {thr:.2f}", f"Step: {step}"]
    lh, pad, bw2 = 22, 8, 150
    by2 = h - ph - 28; by1 = by2 - (lh * len(lines) + pad)
    ov2 = frame.copy()
    cv2.rectangle(ov2, (10, by1), (10 + bw2, by2), COLOR_BLACK, -1)
    cv2.addWeighted(ov2, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (10, by1), (10 + bw2, by2), COLOR_WHITE, 1)
    for i, ln in enumerate(lines):
        cv2.putText(frame, ln, (18, by1 + pad + lh * i + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)

    # GT badge (kanan-atas): cocok? hijau, salah? merah
    if gt_label in (0, 1):
        gt_txt = f"GT: {LABEL_NAMES[gt_label].upper()}"
        ok = (not warming) and (pred == gt_label)
        bcol = COLOR_GREEN if ok else (COLOR_GRAY if warming else COLOR_RED)
        (tw, th_), _ = cv2.getTextSize(gt_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        bx = w - tw - 24
        cv2.rectangle(frame, (bx - 8, 8), (w - 8, 8 + th_ + 28), COLOR_BLACK, -1)
        cv2.putText(frame, gt_txt, (bx, 8 + th_ + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bcol, 2)
        if gt_name:
            cv2.putText(frame, gt_name[:22], (bx, 8 + th_ + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1)
    return frame


# ── Tulis hasil ───────────────────────────────────────────────────────────────

def write_report(out, args, speed, n_proc, fps_video, acc=None):
    """Laporan KECEPATAN (default). acc=dict opsional utk bagian akurasi."""
    L = []
    L.append("=" * 60)
    L.append("KINERJA PIPELINE REALTIME (YOLO11n-pose + BlockGCN)")
    L.append("Video gabungan val — pengukuran end-to-end")
    L.append("=" * 60)
    L.append("Tanggal      : {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    L.append("GPU          : {}".format(args.device))
    L.append("Model        : {}".format(args.weights))
    L.append("YOLO imgsz   : {}".format(args.imgsz))
    L.append("Window/Step  : {} / {}".format(args.window, args.step))
    L.append("Frame diuji  : {}  (~{:.1f} menit @ {:.0f} fps)".format(
        n_proc, n_proc / max(fps_video, 1) / 60, fps_video))
    L.append("")
    L.append("-" * 60)
    L.append("KECEPATAN / KOMPUTASI")
    L.append("-" * 60)
    L.append("  YOLO11n-pose per frame     : {:7.2f} ms".format(speed['yolo_ms']))
    L.append("  BlockGCN per window        : {:7.2f} ms".format(speed['gcn_ms']))
    L.append("  Total pipeline per frame   : {:7.2f} ms   (= YOLO + BlockGCN/step)".format(
        speed['eff_ms']))
    L.append("  Latency deteksi per frame  : {:7.2f} ms".format(speed['eff_ms']))
    L.append("  FPS (kemampuan proses)     : {:7.1f} FPS".format(speed['eff_fps']))
    L.append("  FPS (wall-clock observasi) : {:7.1f} FPS".format(speed['wall_fps']))
    L.append("")
    L.append("Tabel siap-tempel:")
    L.append("  YOLO (ms) | BlockGCN (ms) | Total/frame (ms) | FPS")
    L.append("  {:.2f}     | {:.2f}          | {:.2f}            | {:.1f}".format(
        speed['yolo_ms'], speed['gcn_ms'], speed['eff_ms'], speed['eff_fps']))

    if acc is not None:
        fm, fm_clean, cm_clip, ev = acc['fm'], acc['fm_clean'], acc['cm_clip'], acc['ev']
        L.append("")
        L.append("=" * 60)
        L.append("AKURASI PER-KLIP (agregasi peak P(fall) per video asli)")
        L.append("=" * 60)
        mode = ("buffer di-RESET tiap klip (tanpa carryover, setara per-video)"
                if getattr(args, "reset_on_clip", False)
                else "stream kontinu (ADA carryover antar-klip -> hati-hati)")
        L.append("  Mode              : {}".format(mode))
        L.append("  Total klip        : {}  (not_fall={} fall={})".format(
            cm_clip['n'], cm_clip['TN'] + cm_clip['FP'], cm_clip['TP'] + cm_clip['FN']))
        L.append("  Accuracy          : {:.2f}%".format(cm_clip['acc'] * 100))
        L.append("  Balanced Accuracy : {:.2f}%".format(cm_clip['bal_acc'] * 100))
        L.append("  Precision         : {:.2f}%".format(cm_clip['prec'] * 100))
        L.append("  Sensitivity/Recall: {:.2f}%".format(cm_clip['sens'] * 100))
        L.append("  Specificity       : {:.2f}%".format(cm_clip['spec'] * 100))
        L.append("  F1-Score          : {:.2f}%".format(cm_clip['f1'] * 100))
        L.append("  " + "-" * 48)
        L.append("  Confusion Matrix (per-klip):")
        L.append("                     Pred NOT_FALL  Pred FALL")
        L.append("    True NOT_FALL        {:^5}         {:^5}  (TN / FP)".format(
            cm_clip['TN'], cm_clip['FP']))
        L.append("    True FALL            {:^5}         {:^5}  (FN / TP)".format(
            cm_clip['FN'], cm_clip['TP']))
        L.append("")
        L.append("-" * 60)
        L.append("ANALISIS STREAMING (berkelanjutan)")
        L.append("-" * 60)
        L.append("  Frame-level (bersih) : Acc {:.2f}%  Sens {:.2f}%  Spec {:.2f}%  F1 {:.2f}%".format(
            fm_clean['acc']*100, fm_clean['sens']*100, fm_clean['spec']*100, fm_clean['f1']*100))
        L.append("  Event jatuh          : {}/{} terdeteksi, jeda {:.2f}s, "
                 "false-alarm {:.2f}/menit".format(
                     ev['detected'], ev['n_events'], ev['mean_latency_s'], ev['fa_per_min']))

        if acc.get('clip_rows'):
            L.append("")
            L.append("=" * 60)
            L.append("PREDIKSI PER-KLIP (urutan stream)")
            L.append("=" * 60)
            W = 38
            L.append("# {:<{w}} | {:<10} | {:<10} | {:<7} | peak_fall".format(
                "nama_video", "label_asli", "prediksi", "hasil", w=W))
            L.append("# " + "-" * 84)
            for name, lab, pr, peak in acc['clip_rows']:
                pr_name = LABEL_NAMES.get(pr, "?")
                hasil = "BENAR" if pr == lab else ("-" if pr < 0 else "SALAH")
                L.append("{:<{w}} | {:<10} | {:<10} | {:<7} | {:>8.4f}".format(
                    name, LABEL_NAMES[lab], pr_name, hasil, peak, w=W))

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Eval kinerja realtime di video gabungan val")
    ap.add_argument("--video",     default=DEFAULT_VIDEO)
    ap.add_argument("--timeline",  default=DEFAULT_TIMELINE)
    ap.add_argument("--weights",   default=DEFAULT_WEIGHTS)
    ap.add_argument("--config",    default=DEFAULT_CONFIG)
    ap.add_argument("--yolo",      default=DEFAULT_YOLO)
    ap.add_argument("--out",       default=DEFAULT_OUT)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--window",    type=int, default=DEFAULT_WINDOW)
    ap.add_argument("--step",      type=int, default=DEFAULT_STEP)
    ap.add_argument("--smooth-alpha", type=float, default=DEFAULT_SMOOTH)
    ap.add_argument("--imgsz",     type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--device",    default="cuda:0")
    ap.add_argument("--no-half",   action="store_true")
    ap.add_argument("--max-speed", action="store_true", help="Proses secepatnya (ukur murni)")
    ap.add_argument("--no-display", action="store_true", help="Tanpa jendela (headless)")
    ap.add_argument("--accuracy", action="store_true",
                    help="Hitung juga akurasi/confusion vs timeline (default: hanya kecepatan)")
    ap.add_argument("--reset-on-clip", action="store_true",
                    help="Reset buffer di batas tiap klip (butuh --accuracy). Hilangkan "
                         "carryover antar-video -> akurasi per-klip BERSIH (setara per-video). "
                         "Catatan: metrik streaming (latency/false-alarm) jadi kurang berarti "
                         "di mode ini; untuk itu jalankan TANPA flag ini.")
    ap.add_argument("--width",     type=int, default=1280)
    ap.add_argument("--height",    type=int, default=720)
    ap.add_argument("--shotdir",   default="paper_shots")
    ap.add_argument("--hold-short", type=int, default=500,
                    help="ms menahan frame keputusan klip pendek (<64) di layar biar "
                         "kelihatan/screenshot. 0=tidak menahan. Diabaikan saat --no-display.")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA tidak tersedia, pakai CPU"); args.device = "cpu"
    use_cuda = args.device.startswith("cuda")
    half = use_cuda and not args.no_half
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    video_path = ROOT / args.video if not Path(args.video).is_absolute() else Path(args.video)
    if not video_path.exists():
        sys.exit(f"ERROR: video tidak ditemukan: {video_path}")
    out_abs = str(ROOT / args.out) if not Path(args.out).is_absolute() else args.out

    # Timeline (hanya dipakai bila --accuracy)
    gt = seam = clips = None
    has_gt = False
    if args.accuracy:
        tl_path = Path(args.timeline) if args.timeline else \
            video_path.with_name(video_path.stem + "_timeline.csv")
        if tl_path.exists():
            gt, seam, clips, n_tl = load_timeline(tl_path)
            has_gt = True
            print(f"Timeline GT : {tl_path}  ({len(clips)} klip, {n_tl} frame)")
        else:
            print(f"[WARN] --accuracy diminta tapi timeline tidak ada ({tl_path}) -> "
                  f"hanya ukur kecepatan.")

    # Muat YOLO + model
    from ultralytics import YOLO
    print(f"Memuat YOLO: {args.yolo}")
    yolo = YOLO(args.yolo); yolo.to(args.device)
    with open(ROOT / args.config if not Path(args.config).is_absolute() else args.config) as f:
        cfg = yaml.safe_load(f)
    print(f"Memuat model: {args.weights}")
    model = load_model(cfg, ROOT / args.weights if not Path(args.weights).is_absolute() else args.weights,
                       args.device)
    win = args.window

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"ERROR: tidak bisa membuka video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Video: {total} frame @ {fps_video:.1f} fps  (~{total/fps_video/60:.1f} menit)\n")

    if not args.no_display:
        WIN = "Eval Realtime — Merged Val"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, args.width, args.height)
    shotdir = ROOT / args.shotdir; shotdir.mkdir(parents=True, exist_ok=True)

    # Mode reset-on-clip: butuh timeline
    reset_on_clip = args.reset_on_clip and has_gt
    if args.reset_on_clip and not has_gt:
        print("[WARN] --reset-on-clip butuh --accuracy + timeline. Diabaikan.")
    frame_clip = None
    if reset_on_clip:
        frame_clip = np.full(total, -1, np.int32)
        for ci, c in enumerate(clips):
            frame_clip[c["start"]:min(c["end"] + 1, total)] = ci
        print("Mode RESET-ON-CLIP aktif: buffer di-reset tiap batas klip (tanpa carryover).\n")
    cur_clip = -1
    clip_fc = 0                 # penghitung frame dalam klip berjalan
    clip_has_window = False     # apakah klip berjalan sudah punya >=1 window inferensi
    clip_peak = {}              # idx_klip -> peak P(fall)

    # State
    buf = collections.deque(maxlen=win)
    smoothed = None
    last_pred, last_pf = 0, 0.0
    pred_per_frame = []   # prediksi per frame (0/1/-1 warmup)
    pf_per_frame = []     # P(fall) per frame (-1.0 saat warmup)
    yolo_ms = gcn_ms = 0.0
    n_windows = 0
    fps_times = collections.deque(maxlen=30)
    frame_delay = 1 if args.max_speed else (max(1, int(1000 / fps_video)) if fps_video > 0 else 30)
    fc = 0; ss = 0; paused = False
    t_wall0 = time.time()

    while True:
        if paused and not args.no_display:
            k = cv2.waitKey(30) & 0xFF
            if k == ord('q'): break
            if k == ord(' '): paused = False
            continue

        ok, frame = cap.read()
        if not ok:
            break
        fc += 1
        fidx = fc - 1
        fps_times.append(time.time())
        fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0]) if len(fps_times) >= 2 else 0.0

        # ── Reset buffer di batas klip (mode bersih) ──────────────────────────
        if reset_on_clip:
            cidx = int(frame_clip[fidx]) if fidx < len(frame_clip) else -1
            if cidx != cur_clip:
                # finalisasi klip sebelumnya yg terlalu pendek (belum sempat 1 window)
                if cur_clip >= 0 and not clip_has_window and len(buf) > 0:
                    wnp = np.array(buf)
                    valid = int(((wnp[:, :, 2] > 0.3).sum(axis=1) >= 5).sum())
                    if valid >= win // 4:
                        t1 = time.perf_counter()
                        pf = run_inference(model, win, normalize_skeleton_sequence(wnp), args.device)
                        if use_cuda: torch.cuda.synchronize()
                        gcn_ms += (time.perf_counter() - t1) * 1000.0
                        n_windows += 1
                        clip_peak[cur_clip] = max(clip_peak.get(cur_clip, 0.0), pf)
                # reset state untuk klip baru
                buf.clear(); smoothed = None
                last_pred, last_pf = 0, 0.0
                clip_fc = 0
                clip_has_window = False
                cur_clip = cidx
            clip_fc += 1

        # YOLO
        kf_xy = np.zeros((NUM_JOINTS, 2), np.float32)
        kf_conf = np.zeros(NUM_JOINTS, np.float32)
        t0 = time.perf_counter()
        res = yolo.predict(frame, verbose=False, device=args.device, half=half, imgsz=args.imgsz)
        if use_cuda: torch.cuda.synchronize()
        yolo_ms += (time.perf_counter() - t0) * 1000.0

        person = False
        if res and res[0].keypoints is not None and len(res[0].keypoints) > 0 \
                and res[0].keypoints.conf is not None:
            kps = res[0].keypoints
            ca = kps.conf.cpu().numpy()
            best = int(ca.mean(axis=1).argmax())
            raw_xy = kps.xy.cpu().numpy()[best]
            kf_conf = ca[best]
            if smoothed is None:
                smoothed = raw_xy.copy()
            else:
                hi = kf_conf > 0.3
                smoothed[hi] = args.smooth_alpha * raw_xy[hi] + (1 - args.smooth_alpha) * smoothed[hi]
            kf_xy = raw_xy
            person = True
        if not person:
            smoothed = None

        kf = np.zeros((NUM_JOINTS, 3), np.float32)
        kf[:, :2] = smoothed if smoothed is not None else kf_xy
        kf[:, 2] = kf_conf
        buf.append(kf)

        warming = len(buf) < win
        # Inferensi tiap step frame (counter per-klip bila reset, global bila tidak)
        step_counter = clip_fc if reset_on_clip else fc
        if len(buf) == win and step_counter % args.step == 0:
            wnp = np.array(buf)
            valid = int(((wnp[:, :, 2] > 0.3).sum(axis=1) >= 5).sum())
            if valid < win // 4:
                last_pred, last_pf = 0, 0.0
            else:
                t1 = time.perf_counter()
                pf = run_inference(model, win, normalize_skeleton_sequence(wnp), args.device)
                if use_cuda: torch.cuda.synchronize()
                gcn_ms += (time.perf_counter() - t1) * 1000.0
                n_windows += 1
                last_pf = pf
                last_pred = 1 if pf >= args.threshold else 0
                if reset_on_clip and cur_clip >= 0:
                    clip_peak[cur_clip] = max(clip_peak.get(cur_clip, 0.0), pf)
                    clip_has_window = True

        # Klip PENDEK (<64 frame): buffer tak pernah penuh. Prediksi paksa di frame
        # TERAKHIR klip (pakai semua frame, resample ke 64) supaya tetap ada
        # keputusan -> masuk metrik DAN tampil di layar (bukan "warming up" terus).
        finalized_short = False
        if reset_on_clip and not clip_has_window:
            is_last = (fidx + 1 >= total) or (frame_clip[fidx + 1] != cur_clip)
            if is_last and len(buf) > 0:
                wnp = np.array(buf)
                valid = int(((wnp[:, :, 2] > 0.3).sum(axis=1) >= 5).sum())
                if valid >= win // 4:
                    t1 = time.perf_counter()
                    pf = run_inference(model, win, normalize_skeleton_sequence(wnp), args.device)
                    if use_cuda: torch.cuda.synchronize()
                    gcn_ms += (time.perf_counter() - t1) * 1000.0
                    n_windows += 1
                    last_pf = pf
                    last_pred = 1 if pf >= args.threshold else 0
                    clip_peak[cur_clip] = max(clip_peak.get(cur_clip, 0.0), pf)
                    clip_has_window = True
                    warming = False          # tampilkan keputusan, bukan warming-up
                    finalized_short = True

        pred_per_frame.append(-1 if warming else last_pred)
        pf_per_frame.append(-1.0 if warming else last_pf)

        # Display
        if not args.no_display:
            if person:
                frame = draw_skeleton(frame, kf_xy, kf_conf)
            gl = int(gt[fidx]) if (has_gt and fidx < len(gt)) else -1
            gn = ""
            if has_gt:
                for c in clips:
                    if c["start"] <= fidx <= c["end"]:
                        gn = c["name"]; break
            frame = draw_panel(frame, last_pred, last_pf, warming, len(buf), win,
                               fps, args.threshold, fc, total, args.step, gl, gn)
            h, w = frame.shape[:2]
            sc = min(args.width / w, args.height / h)
            cv2.imshow(WIN, cv2.resize(frame, (int(w * sc), int(h * sc))))
            # tahan sebentar keputusan klip pendek biar kelihatan / bisa di-screenshot
            wait_ms = max(args.hold_short, frame_delay) if finalized_short else frame_delay
            k = cv2.waitKey(wait_ms) & 0xFF
            if k == ord('q'):
                print("\nDihentikan user (hasil parsial disimpan).")
                break
            if k == ord(' '):
                paused = True
            if k == ord('s'):
                p = shotdir / f"merged_{ss:03d}_f{fc}.jpg"; cv2.imwrite(str(p), frame); ss += 1
                print(f"Screenshot: {p}")
        if fc % 500 == 0:
            print(f"\r  [{fc}/{total}] fps_proses~{fps:.1f}", end="", flush=True)

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    print()

    # Finalisasi klip TERAKHIR (mode reset) bila terlalu pendek / belum ada window
    if reset_on_clip and cur_clip >= 0 and not clip_has_window and len(buf) > 0:
        wnp = np.array(buf)
        valid = int(((wnp[:, :, 2] > 0.3).sum(axis=1) >= 5).sum())
        if valid >= win // 4:
            pf = run_inference(model, win, normalize_skeleton_sequence(wnp), args.device)
            n_windows += 1
            clip_peak[cur_clip] = max(clip_peak.get(cur_clip, 0.0), pf)

    wall = time.time() - t_wall0
    n_proc = len(pred_per_frame)
    speed = dict(
        yolo_ms=yolo_ms / max(n_proc, 1),
        gcn_ms=gcn_ms / max(n_windows, 1),
        wall_fps=n_proc / max(wall, 1e-6),
    )
    speed["eff_ms"] = speed["yolo_ms"] + speed["gcn_ms"] / max(args.step, 1)
    speed["eff_fps"] = 1000.0 / max(speed["eff_ms"], 1e-6)

    # ── Akurasi opsional (hanya bila --accuracy + timeline tersedia) ─────────
    acc = None
    if has_gt:
        pred = np.array(pred_per_frame, np.int8)
        m = min(len(pred), len(gt))
        pred = pred[:m]; gt_a = gt[:m]; seam_a = seam[:m]
        valid_mask = (gt_a >= 0) & (pred >= 0)
        fm = confusion(gt_a, pred, valid_mask)
        fm_clean = confusion(gt_a, pred, valid_mask & (~seam_a))

        pf_a = np.array(pf_per_frame, np.float32)[:m]
        TP = TN = FP = FN = 0
        clip_rows = []          # (name, label, pred, peak_score)
        for ci, c in enumerate(clips):
            if reset_on_clip:
                # peak bersih per-klip (tanpa carryover) dari clip_peak
                if ci in clip_peak:
                    peak = float(clip_peak[ci])
                    clip_pred = 1 if peak >= args.threshold else 0
                else:
                    peak, clip_pred = 0.0, -1   # tak ada window valid sama sekali
            else:
                seg = pred[c["start"]:min(c["end"] + 1, m)]
                seg_valid = seg[seg >= 0]
                seg_pf = pf_a[c["start"]:min(c["end"] + 1, m)]
                seg_pf = seg_pf[seg_pf >= 0.0]
                peak = float(seg_pf.max()) if len(seg_pf) else 0.0
                clip_pred = (1 if (seg_valid == 1).any() else 0) if len(seg_valid) else -1
            clip_rows.append((c["name"], c["label"], clip_pred, peak))
            if clip_pred < 0:
                continue
            if c["label"] == 1:
                TP += clip_pred == 1; FN += clip_pred == 0
            else:
                FP += clip_pred == 1; TN += clip_pred == 0
        cm_clip = metrics_from_cm(int(TP), int(TN), int(FP), int(FN))

        fall_clips = [c for c in clips if c["label"] == 1]
        latencies_f, detected = [], 0
        for c in fall_clips:
            seg = pred[c["start"]:min(c["end"] + 1, m)]
            hits = np.where(seg == 1)[0]
            if len(hits) > 0:
                detected += 1; latencies_f.append(int(hits[0]))
        mean_lat_f = float(np.mean(latencies_f)) if latencies_f else 0.0

        nf_frames = fa_episodes = 0; prev_fa = False
        for c in clips:
            if c["label"] != 0:
                prev_fa = False; continue
            seg = pred[c["start"]:min(c["end"] + 1, m)]
            nf_frames += int((seg >= 0).sum())
            for v in seg:
                cur = (v == 1)
                if cur and not prev_fa:
                    fa_episodes += 1
                prev_fa = cur
        nf_min = nf_frames / fps_video / 60.0
        ev = dict(n_events=len(fall_clips), detected=detected,
                  missed=len(fall_clips) - detected,
                  mean_latency_s=mean_lat_f / fps_video,
                  fa_per_min=fa_episodes / max(nf_min, 1e-6))
        acc = dict(fm=fm, fm_clean=fm_clean, cm_clip=cm_clip, ev=ev,
                   clip_rows=clip_rows)

    write_report(out_abs, args, speed, n_proc, fps_video, acc)

    print("=" * 60)
    print("  HASIL — Kinerja Realtime di Video Gabungan Val")
    print("=" * 60)
    print(f"  YOLO11n-pose per frame   : {speed['yolo_ms']:.2f} ms")
    print(f"  BlockGCN per window      : {speed['gcn_ms']:.2f} ms")
    print(f"  Total pipeline per frame : {speed['eff_ms']:.2f} ms")
    print(f"  FPS (kemampuan proses)   : {speed['eff_fps']:.1f} FPS")
    print(f"  FPS (wall-clock)         : {speed['wall_fps']:.1f} FPS")
    print("=" * 60)
    if acc is not None:
        cm = acc['cm_clip']
        print("  AKURASI PER-KLIP (agregasi peak P(fall)):")
        print(f"    Total klip        : {cm['n']}  (not_fall={cm['TN']+cm['FP']} fall={cm['TP']+cm['FN']})")
        print(f"    Accuracy          : {cm['acc']*100:.2f}%")
        print(f"    Balanced Accuracy : {cm['bal_acc']*100:.2f}%")
        print(f"    Precision         : {cm['prec']*100:.2f}%")
        print(f"    Sensitivity/Recall: {cm['sens']*100:.2f}%")
        print(f"    Specificity       : {cm['spec']*100:.2f}%")
        print(f"    F1-Score          : {cm['f1']*100:.2f}%")
        print("    Confusion Matrix:")
        print("                       Pred NOT_FALL  Pred FALL")
        print(f"      True NOT_FALL        {cm['TN']:^5}         {cm['FP']:^5}")
        print(f"      True FALL            {cm['FN']:^5}         {cm['TP']:^5}")
        print(f"    [streaming] frame bersih Acc {acc['fm_clean']['acc']*100:.1f}% | "
              f"event {acc['ev']['detected']}/{acc['ev']['n_events']} "
              f"jeda {acc['ev']['mean_latency_s']:.2f}s")
        print("=" * 60)
    print(f"\n  Tersimpan: {out_abs}")


if __name__ == "__main__":
    main()