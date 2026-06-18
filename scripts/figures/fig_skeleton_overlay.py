"""
fig_skeleton_overlay.py
=======================
Contoh ekstraksi skeleton YOLO11-pose untuk paper.
Menghasilkan 2 panel per kasus:
    kiri  = frame asli + overlay skeleton
    kanan = skeleton di atas kanvas PUTIH (nomor sendi + sendi conf-rendah ditandai merah)

Auto-pilih dari video:
    - frame BERHASIL : rata-rata confidence tertinggi (subjek terlihat utuh)
    - frame GAGAL    : ada >= 3 sendi confidence < 0,4 (oklusi / terpotong)

Output (docs/figures_paper/):
    fig_ekstraksi_berhasil.png    ← GAMBAR_EKSTRAKSI_BERHASIL
    fig_ekstraksi_gagal.png       ← GAMBAR_EKSTRAKSI_GAGAL

Jalankan (contoh video fall NTU):
    conda activate block-gcn
    python scripts/fig_skeleton_overlay.py --video dataset/ntu_videos/S001C001P001R001A043_rgb.avi
    # ganti --video untuk frame berhasil yang lebih jelas (mis. A008 sitting):
    python scripts/fig_skeleton_overlay.py --good-video dataset/ntu_videos/S001C001P001R001A008_rgb.avi \
                                           --bad-video  dataset/ntu_videos/S001C001P001R001A043_rgb.avi
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("Install: pip install ultralytics opencv-python")

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "docs/figures_paper"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "savefig.facecolor": "white", "figure.facecolor": "white"})

WEIGHTS = ROOT / "yolo11n-pose.pt"
NAMES = ["hidung", "mata-ki", "mata-ka", "telinga-ki", "telinga-ka", "bahu-ki", "bahu-ka",
         "siku-ki", "siku-ka", "tangan-ki", "tangan-ka", "pinggul-ki", "pinggul-ka",
         "lutut-ki", "lutut-ka", "kaki-ki", "kaki-ka"]
EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
         (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
CONF_LOW = 0.40


def bad_joint_count(kp, W, H):
    """Sendi dianggap buruk bila conf rendah ATAU keluar bidang frame (terpotong)."""
    n = 0
    for i in range(17):
        x, y, c = kp[i]
        out = (x < 2 or x > W - 2 or y < 2 or y > H - 2)
        if c < CONF_LOW or out:
            n += 1
    return n


def scan_video(model, path, want="good", best=None):
    """Scan satu video, update `best` (skor, frame, kpts). Kembalikan best."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  [SKIP] gagal buka {path}")
        return best
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % 2 == 0:   # subsample untuk cepat
            continue
        H, W = frame.shape[:2]
        res = model(frame, verbose=False)[0]
        if res.keypoints is None or len(res.keypoints) == 0:
            continue
        karr = res.keypoints.data.cpu().numpy()  # (n,17,3)
        confs = karr[:, :, 2].mean(axis=1)
        j = int(confs.argmax())
        kp = karr[j]
        mean_conf = float(kp[:, 2].mean())
        n_bad = bad_joint_count(kp, W, H)
        if want == "good":
            score = mean_conf - 0.5 * n_bad
        else:
            # bad: maks sendi buruk; tie-break conf rendah. Orang tetap terdeteksi.
            score = n_bad + (1.0 - mean_conf)
        if best is None or score > best[0]:
            best = (score, frame.copy(), kp)
    cap.release()
    return best


def draw_panel(frame_bgr, kp, title, out_path):
    """Gambar skeleton langsung di frame, simpan full-res (tanpa judul/panel/border)."""
    img = frame_bgr.copy()
    H, W = img.shape[:2]
    lw = max(2, round(W / 480))          # tebal garis adaptif
    r = max(3, round(W / 200))           # radius sendi adaptif
    CYAN = (255, 255, 0)                 # BGR
    GREEN = (0, 220, 70)
    RED = (0, 0, 220)

    for a, b in EDGES:
        if kp[a, 2] > 0.05 and kp[b, 2] > 0.05:
            pa = (int(kp[a, 0]), int(kp[a, 1]))
            pb = (int(kp[b, 0]), int(kp[b, 1]))
            cv2.line(img, pa, pb, CYAN, lw, cv2.LINE_AA)
    for i in range(17):
        if kp[i, 2] <= 0.05:
            continue
        c = RED if kp[i, 2] < CONF_LOW else GREEN
        p = (int(kp[i, 0]), int(kp[i, 1]))
        cv2.circle(img, p, r, c, -1, cv2.LINE_AA)
        cv2.circle(img, p, r, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)
    n_low = int((kp[:, 2] < CONF_LOW).sum())
    print(f"  -> {out_path}  (mean_conf={kp[:,2].mean():.2f}, sendi_rendah={n_low})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="satu video; dipakai utk good & bad bila --good/--bad tak diisi")
    ap.add_argument("--good-video", help="video untuk contoh BERHASIL")
    ap.add_argument("--bad-video", help="video untuk contoh GAGAL/oklusi")
    ap.add_argument("--bad-glob", help="pola banyak video utk cari frame TERBURUK global, "
                                       "mis. 'dataset/ntu_videos/*A043*.avi'")
    ap.add_argument("--bad-limit", type=int, default=15, help="maks video discan utk --bad-glob")
    args = ap.parse_args()

    good_v = args.good_video or args.video
    if not good_v:
        sys.exit("Beri --video ATAU --good-video")

    if not WEIGHTS.exists():
        sys.exit(f"Bobot YOLO tidak ada: {WEIGHTS}")
    model = YOLO(str(WEIGHTS))

    print("=== Cari frame BERHASIL ===")
    best_good = scan_video(model, good_v, want="good")
    if best_good is None:
        sys.exit(f"Tidak ada pose terdeteksi di {good_v}")
    draw_panel(best_good[1], best_good[2], "Contoh Ekstraksi Skeleton BERHASIL (kondisi ideal)",
               OUT / "fig_ekstraksi_berhasil.png")

    print("=== Cari frame GAGAL/oklusi/terpotong ===")
    if args.bad_glob:
        import glob
        vids = sorted(glob.glob(str(ROOT / args.bad_glob)))[: args.bad_limit]
        if not vids:
            sys.exit(f"--bad-glob tak cocok: {args.bad_glob}")
        print(f"  scan {len(vids)} video...")
        best_bad = None
        for v in vids:
            best_bad = scan_video(model, v, want="bad", best=best_bad)
    else:
        bad_v = args.bad_video or args.video
        if not bad_v:
            sys.exit("Beri --bad-video atau --bad-glob")
        best_bad = scan_video(model, bad_v, want="bad")
    if best_bad is None:
        sys.exit("Tidak ada pose terdeteksi utk kasus gagal")
    draw_panel(best_bad[1], best_bad[2], "Contoh Ekstraksi Skeleton PARSIAL (oklusi / terpotong)",
               OUT / "fig_ekstraksi_gagal.png")
    print("Selesai.")


if __name__ == "__main__":
    main()
