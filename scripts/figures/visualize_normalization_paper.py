"""
visualize_normalization_paper.py
================================
Versi *light-mode*, siap-cetak (paper) dari visualize_normalization.py.
Latar putih, font serif, DPI 300, palet ramah grayscale.

Output (docs/figures_paper/):
    fig_normalisasi_singleframe.png
    fig_normalisasi_multiframe.png
    fig_fall_vs_notfall.png   (jika file not_fall tersedia)

Cara pakai:
    python scripts/visualize_normalization_paper.py
    python scripts/visualize_normalization_paper.py --npy dataset/ntu_skeleton/fall/S001C001P001R001A043_rgb.npy
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.extract_skeleton import normalize_skeleton, denormalize_skeleton  # noqa: E402

# ── Style paper (light-mode) ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "figure.dpi": 300,
})

FACE = "white"        # latar figure & axes
EDGE = "#333333"      # garis tulang
SPINE = "#999999"     # garis bingkai axes
GRID = "#dddddd"

# ── COCO 17-joint skeleton edges ──────────────────────────────────────────────
SKELETON_EDGES = [
    (0,1),(0,2),(1,3),(2,4),
    (0,5),(0,6),(5,6),
    (5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

# Warna per region — dipilih agar tetap terbaca saat dicetak grayscale
JOINT_COLORS = {
    "head":  [0,1,2,3,4],
    "torso": [5,6,11,12],
    "arm":   [7,8,9,10],
    "leg":   [13,14,15,16],
}
COLOR_MAP = {}
for j in JOINT_COLORS["head"]:  COLOR_MAP[j] = "#d62728"   # merah
for j in JOINT_COLORS["torso"]: COLOR_MAP[j] = "#1f77b4"   # biru
for j in JOINT_COLORS["arm"]:   COLOR_MAP[j] = "#2ca02c"   # hijau
for j in JOINT_COLORS["leg"]:   COLOR_MAP[j] = "#ff7f0e"   # oranye


def draw_skeleton(ax, frame, title, conf_thresh=0.2, show_labels=False):
    """Gambar satu frame skeleton pada axes (light-mode)."""
    xy   = frame[:, :2]
    conf = frame[:, 2]

    # Flip sumbu Y agar kepala di atas (koordinat citra: Y ke bawah)
    y_vals = -xy[:, 1]

    for i, j in SKELETON_EDGES:
        if conf[i] > conf_thresh and conf[j] > conf_thresh:
            ax.plot([xy[i, 0], xy[j, 0]],
                    [y_vals[i], y_vals[j]],
                    color=EDGE, linewidth=1.6, zorder=1)

    for k in range(17):
        if conf[k] > conf_thresh:
            ax.scatter(xy[k, 0], y_vals[k], s=42,
                       color=COLOR_MAP.get(k, "#000000"),
                       edgecolors=EDGE, linewidths=0.5, zorder=2)
            if show_labels:
                ax.annotate(f"{k}", (xy[k, 0], y_vals[k]),
                            textcoords="offset points", xytext=(4, 4),
                            fontsize=6, color="#333333")

    ax.set_title(title, fontsize=10, fontweight="bold", pad=8, color="black")
    ax.set_aspect("equal")
    ax.set_facecolor(FACE)
    ax.tick_params(colors="#555555", labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE)


# ── Gambar 1: Single frame perbandingan ──────────────────────────────────────

def plot_single_frame_comparison(seq_raw, seq_norm, frame_idx, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5), facecolor=FACE)

    frame_r = seq_raw[frame_idx]
    frame_n = seq_norm[frame_idx]

    draw_skeleton(axes[0], frame_r, "Sebelum Normalisasi")

    hc_x = (frame_r[11, 0] + frame_r[12, 0]) / 2
    hc_y = -((frame_r[11, 1] + frame_r[12, 1]) / 2)
    axes[0].scatter(hc_x, hc_y, s=110, marker="+",
                    color="#d62728", linewidths=2, zorder=5, label="Titik tengah pinggul")
    axes[0].legend(loc="lower right", fontsize=8, framealpha=0.9,
                   facecolor="white", edgecolor=SPINE, labelcolor="black")

    draw_skeleton(axes[1], frame_n, "Sesudah Normalisasi")

    axes[1].scatter(0, 0, s=110, marker="+",
                    color="#d62728", linewidths=2, zorder=5, label="Origin")
    axes[1].axhline(0, color="#d62728", linewidth=0.6, linestyle="--", alpha=0.5)
    axes[1].axvline(0, color="#d62728", linewidth=0.6, linestyle="--", alpha=0.5)
    axes[1].legend(loc="lower right", fontsize=8, framealpha=0.9,
                   facecolor="white", edgecolor=SPINE, labelcolor="black")

    legend_items = [
        mpatches.Patch(color="#d62728", label="Kepala"),
        mpatches.Patch(color="#1f77b4", label="Torso"),
        mpatches.Patch(color="#2ca02c", label="Lengan"),
        mpatches.Patch(color="#ff7f0e", label="Kaki"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=4,
               facecolor="white", edgecolor=SPINE, labelcolor="black",
               fontsize=9, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=FACE)
    print(f"Disimpan: {out_path}")
    plt.close()


# ── Gambar 2: Multi-frame sequence (sebelum vs sesudah) ───────────────────────

def plot_multiframe_comparison(seq_raw, seq_norm, out_path, n_frames=6):
    total_valid = int(((seq_raw[:, :, 2] > 0.2).sum(axis=1) >= 3).sum())
    indices = np.linspace(0, max(total_valid - 1, 0), n_frames, dtype=int)

    fig, axes = plt.subplots(2, n_frames,
                             figsize=(2.6 * n_frames, 5.6), facecolor=FACE)

    for col, fi in enumerate(indices):
        draw_skeleton(axes[0, col], seq_raw[fi], f"Frame {fi}")
        draw_skeleton(axes[1, col], seq_norm[fi], f"Frame {fi}")
        axes[1, col].axhline(0, color="#d62728", linewidth=0.6,
                             linestyle="--", alpha=0.5)
        axes[1, col].axvline(0, color="#d62728", linewidth=0.6,
                             linestyle="--", alpha=0.5)

    axes[0, 0].set_ylabel("Sebelum\nNormalisasi", color="black",
                          fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Sesudah\nNormalisasi", color="black",
                          fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=FACE)
    print(f"Disimpan: {out_path}")
    plt.close()


# ── Gambar 3: Dua kelas berdampingan (fall vs not_fall) ───────────────────────

def plot_class_comparison(fall_seq, nf_seq, out_path):
    fi_fall = len(fall_seq) // 2
    fi_nf   = len(nf_seq) // 2

    fall_norm, _, _ = normalize_skeleton(fall_seq)
    nf_norm,   _, _ = normalize_skeleton(nf_seq)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2), facecolor=FACE)

    draw_skeleton(axes[0], fall_norm[fi_fall],
                  "Kelas Fall\nA043 — falling down")
    draw_skeleton(axes[1], nf_norm[fi_nf],
                  "Kelas Not Fall\nA008 — sitting down")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=FACE)
    print(f"Disimpan: {out_path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy",
        default="dataset/ntu_skeleton/fall/S001C001P001R001A043_rgb.npy")
    ap.add_argument("--npy_nf",
        default="dataset/ntu_skeleton/not_fall/S001C001P001R001A008_rgb.npy")
    ap.add_argument("--frame_idx", type=int, default=None)
    ap.add_argument("--outdir", default="docs/figures_paper")
    args = ap.parse_args()

    npy_path = Path(ROOT / args.npy)
    npy_nf   = Path(ROOT / args.npy_nf)
    outdir   = Path(ROOT / args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not npy_path.exists():
        sys.exit(f"File tidak ditemukan: {npy_path}")

    seq_norm = np.load(str(npy_path)).astype(np.float32)   # (T, 17, 3) sudah dinorm

    meta_path = Path(str(npy_path).replace(".npy", "_meta.npz"))
    if meta_path.exists():
        meta       = np.load(str(meta_path))
        sc         = float(meta["sc"])
        hc         = meta["hc"].astype(np.float32)
        seq_unnorm = denormalize_skeleton(seq_norm, sc, hc)
        print(f"Metadata: sc={sc:.2f}px, hc_mean={hc.mean(axis=0)}")
    else:
        print("[WARN] _meta.npz tidak ditemukan — pakai simulasi piksel (sc=200, center=320,240)")
        seq_unnorm = seq_norm.copy()
        seq_unnorm[:, :, :2] = seq_norm[:, :, :2] * 200 + np.array([320, 240])

    seq_norm_recalc, _, _ = normalize_skeleton(seq_unnorm)

    print(f"File  : {npy_path.name}")
    print(f"Shape : {seq_norm.shape}  (T={seq_norm.shape[0]}, V=17, C=3)")

    valid_counts = (seq_unnorm[:, :, 2] > 0.2).sum(axis=1)
    best_frame   = int(valid_counts.argmax()) if args.frame_idx is None else args.frame_idx
    print(f"Frame dipilih: {best_frame} ({int(valid_counts[best_frame])} joint valid)")

    plot_single_frame_comparison(
        seq_unnorm, seq_norm_recalc, best_frame,
        outdir / "fig_normalisasi_singleframe.png"
    )
    plot_multiframe_comparison(
        seq_unnorm, seq_norm_recalc,
        outdir / "fig_normalisasi_multiframe.png",
        n_frames=6
    )

    if npy_nf.exists():
        nf_norm = np.load(str(npy_nf)).astype(np.float32)
        nf_meta = Path(str(npy_nf).replace(".npy", "_meta.npz"))
        if nf_meta.exists():
            m2        = np.load(str(nf_meta))
            nf_unnorm = denormalize_skeleton(nf_norm, float(m2["sc"]), m2["hc"].astype(np.float32))
        else:
            nf_unnorm = nf_norm.copy()
            nf_unnorm[:, :, :2] = nf_norm[:, :, :2] * 200 + np.array([320, 240])
        plot_class_comparison(
            seq_unnorm, nf_unnorm,
            outdir / "fig_fall_vs_notfall.png"
        )
    else:
        print(f"[SKIP] File not_fall tidak ditemukan: {npy_nf}")

    print("\nSelesai. File gambar tersimpan di:", outdir)


if __name__ == "__main__":
    main()
