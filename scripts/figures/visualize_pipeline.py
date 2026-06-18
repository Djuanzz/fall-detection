"""
visualize_pipeline.py
=====================
Flowchart pipeline ekstraksi skeleton dan persiapan dataset.

Output: fig_pipeline_ekstraksi.png

Cara pakai:
    python scripts/visualize_pipeline.py
    python scripts/visualize_pipeline.py --outdir docs/figures
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BG      = "#0d1117"
C_PROC  = "#1a3a5c"   # kotak proses — biru tua
C_IO    = "#1a4a2e"   # kotak I/O — hijau tua
C_DEC   = "#4a2e1a"   # diamond keputusan — coklat tua
C_OUT   = "#3a1a4a"   # output akhir — ungu tua
BORDER  = "#4a9eff"
BORDER_IO   = "#2ecc71"
BORDER_DEC  = "#f39c12"
BORDER_OUT  = "#9b59b6"
TEXT    = "white"
ARROW   = "#888888"


def draw_box(ax, cx, cy, w, h, label, sublabel=None,
             fc=C_PROC, ec=BORDER, fontsize=9):
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                         boxstyle="round,pad=0.02",
                         facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=3)
    ax.add_patch(box)
    if sublabel:
        ax.text(cx, cy + h*0.12, label, ha="center", va="center",
                color=TEXT, fontsize=fontsize, fontweight="bold", zorder=4)
        ax.text(cx, cy - h*0.22, sublabel, ha="center", va="center",
                color="#aaaaaa", fontsize=fontsize - 1.5, zorder=4)
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                color=TEXT, fontsize=fontsize, fontweight="bold", zorder=4)


def draw_diamond(ax, cx, cy, w, h, label, sublabel=None,
                 fc=C_DEC, ec=BORDER_DEC, fontsize=8.5):
    diamond = plt.Polygon(
        [[cx, cy + h/2], [cx + w/2, cy], [cx, cy - h/2], [cx - w/2, cy]],
        closed=True, facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=3)
    ax.add_patch(diamond)
    if sublabel:
        ax.text(cx, cy + h*0.1, label, ha="center", va="center",
                color=TEXT, fontsize=fontsize, fontweight="bold", zorder=4)
        ax.text(cx, cy - h*0.2, sublabel, ha="center", va="center",
                color="#aaaaaa", fontsize=fontsize - 1.5, zorder=4)
    else:
        ax.text(cx, cy, label, ha="center", va="center",
                color=TEXT, fontsize=fontsize, fontweight="bold", zorder=4)


def arrow(ax, x1, y1, x2, y2, label=None, color=ARROW):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.5, mutation_scale=14),
                zorder=2)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.05, my, label, color="#cccccc", fontsize=7.5, zorder=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="docs/figures")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 14), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-1, 8)
    ax.set_ylim(-0.5, 16.5)
    ax.axis("off")

    fig.suptitle("Pipeline Ekstraksi Skeleton & Persiapan Dataset",
                 fontsize=12, color="white", fontweight="bold", y=0.99)

    cx = 3.5   # center x
    W  = 4.0   # lebar kotak
    H  = 0.72  # tinggi kotak
    WD = 3.0   # lebar diamond
    HD = 0.85  # tinggi diamond
    GAP = 1.35 # jarak antar node

    # ── Stage 1 label ────────────────────────────────────────────────────────
    ax.text(0.1, 15.8, "STAGE 1: Ekstraksi Skeleton",
            color=BORDER_IO, fontsize=8, fontweight="bold")
    ax.axhline(15.65, xmin=0.01, xmax=0.99, color=BORDER_IO,
               linewidth=0.5, linestyle="--", alpha=0.4)

    # Node y-positions (top → bottom)
    y = [15.2, 13.85, 12.5, 11.2, 9.7, 8.55, 7.4]

    # 1. Input video
    draw_box(ax, cx, y[0], W, H,
             "Video NTU RGB+D",
             sublabel=".avi  |  A043 (fall), A008/A009/A027/A042 (not_fall)",
             fc=C_IO, ec=BORDER_IO)

    # 2. YOLO
    draw_box(ax, cx, y[1], W, H,
             "YOLO11n-pose",
             sublabel="17 keypoint COCO per frame  |  confidence per joint",
             fc=C_PROC, ec=BORDER)

    # 3. Seleksi subjek
    draw_box(ax, cx, y[2], W, H,
             "Seleksi Subjek Terbaik",
             sublabel="argmax mean confidence  →  1 orang per frame",
             fc=C_PROC, ec=BORDER)

    # 4. Filter diamond
    draw_diamond(ax, cx, y[3], WD, HD,
                 "≥ 40% frame valid?",
                 sublabel="(conf > 0.2 minimal 1 joint)")

    # 4a. GAGAL → keluar kiri
    ax.text(cx - WD/2 - 0.1, y[3], "Tidak", color=BORDER_DEC,
            fontsize=8, ha="right", va="center")
    ax.annotate("", xy=(cx - WD/2 - 1.0, y[3]),
                xytext=(cx - WD/2, y[3]),
                arrowprops=dict(arrowstyle="-|>", color=BORDER_DEC,
                                lw=1.2, mutation_scale=12), zorder=2)
    draw_box(ax, cx - WD/2 - 1.6, y[3], 1.0, 0.55,
             "Skip", fc="#3a1a1a", ec=BORDER_DEC, fontsize=8)

    # 4b. Ya → lanjut bawah
    ax.text(cx + 0.12, (y[3] + y[4]) / 2, "Ya", color=BORDER_IO,
            fontsize=8, va="center")

    # 5. Normalisasi
    draw_box(ax, cx, y[4], W, H,
             "Normalisasi",
             sublabel="hip-center translation  +  shoulder-width scale",
             fc=C_PROC, ec=BORDER)

    # 6. Simpan .npy
    draw_box(ax, cx, y[5], W, H,
             "Simpan .npy",
             sublabel="shape (T, 17, 3)  |  fall/ atau not_fall/",
             fc=C_IO, ec=BORDER_IO)

    # ── Stage 2 label ────────────────────────────────────────────────────────
    ax.text(0.1, 7.0, "STAGE 2: Persiapan Dataset",
            color="#9b59b6", fontsize=8, fontweight="bold")
    ax.axhline(6.85, xmin=0.01, xmax=0.99, color="#9b59b6",
               linewidth=0.5, linestyle="--", alpha=0.4)

    # 7. prepare_dataset
    draw_box(ax, cx, y[6], W, H,
             "prepare_dataset_yolo17.py",
             sublabel="cross-subject split  |  3 modality: joint / bone / motion",
             fc=C_PROC, ec=BORDER)

    # 8. Split balanced / full
    y8 = 6.1
    draw_diamond(ax, cx, y8, WD, HD,
                 "Konfigurasi Dataset",
                 sublabel="balanced  vs  full")

    # Balanced kiri
    xL = cx - 2.2
    draw_box(ax, xL, 4.9, 2.2, H,
             "Balanced",
             sublabel="undersample not_fall\n1 : 1  (631 : 631)",
             fc=C_PROC, ec=BORDER, fontsize=8)

    # Full kanan
    xR = cx + 2.2
    draw_box(ax, xR, 4.9, 2.2, H,
             "Full",
             sublabel="semua data\npos_weight = 4.14",
             fc=C_PROC, ec=BORDER, fontsize=8)

    # Arrow dari diamond ke balanced/full
    ax.annotate("", xy=(xL, 4.9 + H/2),
                xytext=(cx - WD/2, y8),
                arrowprops=dict(arrowstyle="-|>", color=ARROW,
                                lw=1.2, mutation_scale=12), zorder=2)
    ax.annotate("", xy=(xR, 4.9 + H/2),
                xytext=(cx + WD/2, y8),
                arrowprops=dict(arrowstyle="-|>", color=ARROW,
                                lw=1.2, mutation_scale=12), zorder=2)

    # Gabung ke output akhir
    y_out = 3.6
    ax.annotate("", xy=(cx, y_out + H/2),
                xytext=(xL, 4.9 - H/2),
                arrowprops=dict(arrowstyle="-|>", color=ARROW,
                                lw=1.2, mutation_scale=12), zorder=2)
    ax.annotate("", xy=(cx, y_out + H/2),
                xytext=(xR, 4.9 - H/2),
                arrowprops=dict(arrowstyle="-|>", color=ARROW,
                                lw=1.2, mutation_scale=12), zorder=2)

    # 9. Output tensor
    draw_box(ax, cx, y_out, W, H,
             "train_data.npy  +  train_label.pkl",
             sublabel="Tensor  (N, 3, T, 17, 1)  —  siap untuk training BlockGCN",
             fc=C_OUT, ec=BORDER_OUT)

    # ── Arrows vertikal utama ─────────────────────────────────────────────────
    pairs = [(y[0], y[1]), (y[1], y[2]), (y[2], y[3]),
             (y[3], y[4]), (y[4], y[5]), (y[5], y[6]),
             (y[6], y8), (y8, 6.1 - HD/2)]
    for ya, yb in pairs:
        arrow(ax, cx, ya - H/2 if ya != y[3] else ya - HD/2,
                  cx, yb + H/2 if yb != y8 else yb + HD/2)

    # arrow dari diamond stage2 ke bawah sudah digantikan cabang L/R
    # arrow dari y[6] ke y8
    arrow(ax, cx, y[6] - H/2, cx, y8 + HD/2)

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    out = outdir / "fig_pipeline_ekstraksi.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Disimpan: {out}")
    plt.close()


if __name__ == "__main__":
    main()
