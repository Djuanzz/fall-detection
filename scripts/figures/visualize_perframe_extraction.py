"""
visualize_perframe_extraction.py
================================
Buat gambar ekstraksi skeleton PER FRAME, masing-masing disimpan terpisah.
Tiap frame menampilkan satu tahap pose yang jelas berbeda:
    berdiri  -> membungkuk -> jatuh -> terlentang

Gaya mengikuti fig_normalisasi_multiframe.png (light-mode, siap cetak):
tiap gambar berisi 2 panel — Sebelum vs Sesudah Normalisasi — untuk satu frame.

Output (docs/figure_skeleton/):
    frame_00_berdiri.png
    frame_45_membungkuk.png
    frame_54_jatuh.png
    frame_66_terlentang.png
    (atau sesuai --frames / --labels)

Skeleton High vs Oklusi:
    high
    paper_shots/skeleton/duduk/tinggi/hi_018_f76_c0.97.png -> duduk
    paper_shots/skeleton/berdiri/tinggi/hi_013_f56_c0.97.png -> berdiri
    paper_shots/skeleton/jatuh/tinggi/hi_013_f56_c0.96.png -> jatuh
    paper_shots/skeleton/lompat/tinggi/hi_006_f28_c0.97.png -> lompat
    paper_shots/skeleton/sempoyongan/tinggi/hi_021_f88_c0.97.png -> sempoyongan

    ocl
    paper_shots/skeleton/duduk_oklusi/oklusi/occ_016_f42_low4_miss0.png -> duduk
    paper_shots/skeleton/berdiri_oklusi/oklusi/occ_010_f36_low4_miss0.png -> berdiri
    paper_shots/skeleton/jatuh_oklusi/oklusi/occ_007_f52_low12_miss0.png -> jatuh
    paper_shots/skeleton/lompat_oklusi/oklusi/occ_017_f38_low4_miss0.png -> lompat
    paper_shots/skeleton/sempoyongan_oklusi/oklusi/occ_029_f74_low5_miss0.png -> sempoyongan



Cara pakai:
    python scripts/visualize_perframe_extraction.py
    python scripts/visualize_perframe_extraction.py --frames 0,45,54,66 --labels berdiri,membungkuk,jatuh,terlentang
    python scripts/visualize_perframe_extraction.py --npy dataset/ntu_skeleton/fall/S001C001P001R001A043_rgb.npy
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

# ── Style paper (light-mode) ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "figure.dpi": 300,
})

FACE = "white"
EDGE = "#333333"
SPINE = "#999999"
GRID = "#dddddd"

# ── COCO 17-joint skeleton edges ──────────────────────────────────────────────
SKELETON_EDGES = [
    (0,1),(0,2),(1,3),(2,4),
    (0,5),(0,6),(5,6),
    (5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

JOINT_COLORS = {
    "head":  [0,1,2,3,4],
    "torso": [5,6,11,12],
    "arm":   [7,8,9,10],
    "leg":   [13,14,15,16],
}
COLOR_MAP = {}
for j in JOINT_COLORS["head"]:  COLOR_MAP[j] = "#d62728"
for j in JOINT_COLORS["torso"]: COLOR_MAP[j] = "#1f77b4"
for j in JOINT_COLORS["arm"]:   COLOR_MAP[j] = "#2ca02c"
for j in JOINT_COLORS["leg"]:   COLOR_MAP[j] = "#ff7f0e"


# ── Normalisasi (inline, hindari import ultralytics) ──────────────────────────

def normalize_skeleton(sk):
    sk = sk.copy()
    xy = sk[:, :, :2]
    hc = (xy[:, 11] + xy[:, 12]) / 2.0
    xy -= hc[:, np.newaxis, :]
    d = np.linalg.norm(xy[:, 5] - xy[:, 6], axis=1)
    sc = float(d[d > 1e-5].mean()) if (d > 1e-5).any() else 1.0
    if sc > 1e-5:
        xy /= sc
    sk[:, :, :2] = xy
    return sk, sc, hc


def denormalize_skeleton(sk_norm, sc, hc):
    sk = sk_norm.copy()
    sk[:, :, :2] = sk[:, :, :2] * sc + hc[:, np.newaxis, :]
    return sk


def draw_skeleton(ax, frame, conf_thresh=0.2):
    xy = frame[:, :2]
    conf = frame[:, 2]
    y_vals = -xy[:, 1]

    for i, j in SKELETON_EDGES:
        if conf[i] > conf_thresh and conf[j] > conf_thresh:
            ax.plot([xy[i, 0], xy[j, 0]],
                    [y_vals[i], y_vals[j]],
                    color=EDGE, linewidth=3.0, zorder=1)

    for k in range(17):
        if conf[k] > conf_thresh:
            ax.scatter(xy[k, 0], y_vals[k], s=90,
                       color=COLOR_MAP.get(k, "#000000"),
                       edgecolors=EDGE, linewidths=1.0, zorder=2)

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE)
        spine.set_linewidth(1.0)
        spine.set_visible(True)


def plot_one_frame(seq, frame_idx, out_path):
    """Satu file = satu frame skeleton, ada border, tanpa koordinat/judul/legend."""
    fig, ax = plt.subplots(figsize=(3.2, 5.0), facecolor=FACE)
    draw_skeleton(ax, seq[frame_idx])
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=FACE)
    print(f"Disimpan: {out_path}")
    plt.close()


def auto_select_frames(seq_unnorm, seq_norm, n=6):
    """Pilih n frame yang posenya paling jelas berbeda secara visual.

    Farthest-point sampling pada vektor pose ternormalisasi: mulai dari
    frame pertama yang valid, lalu berulang tambah frame yang paling
    'jauh' (beda pose) dari frame yang sudah terpilih. Hasilnya tiap
    gambar terlihat beda pose, tanpa perlu label manual.
    """
    conf = seq_unnorm[:, :, 2]
    valid = (conf > 0.2).sum(axis=1)
    idx = np.where(valid >= 8)[0]
    if len(idx) == 0:
        idx = np.arange(seq_unnorm.shape[0])

    feats = seq_norm[idx][:, :, :2].reshape(len(idx), -1)  # (M, 34) pose ternormalisasi

    picked_local = [0]  # frame valid paling awal (pose berdiri)
    mind = np.linalg.norm(feats - feats[0], axis=1)
    while len(picked_local) < min(n, len(idx)):
        nxt = int(np.argmax(mind))
        if nxt in picked_local:
            break
        picked_local.append(nxt)
        mind = np.minimum(mind, np.linalg.norm(feats - feats[nxt], axis=1))

    return sorted(int(idx[i]) for i in picked_local)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy",
        default="dataset/ntu_skeleton/fall/S001C001P001R002A043_rgb.npy")
    ap.add_argument("--frames", default=None,
        help="indeks frame dipisah koma, mis. 0,45,54,66 (kosong=auto)")
    ap.add_argument("--n", type=int, default=6,
        help="jumlah frame saat auto-pilih")
    ap.add_argument("--outdir", default="docs/figure_skeleton")
    args = ap.parse_args()

    npy_path = Path(ROOT / args.npy)
    outdir = Path(ROOT / args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not npy_path.exists():
        sys.exit(f"File tidak ditemukan: {npy_path}")

    seq_norm = np.load(str(npy_path)).astype(np.float32)

    meta_path = Path(str(npy_path).replace(".npy", "_meta.npz"))
    if meta_path.exists():
        meta = np.load(str(meta_path))
        seq_unnorm = denormalize_skeleton(
            seq_norm, float(meta["sc"]), meta["hc"].astype(np.float32))
        print(f"Metadata: sc={float(meta['sc']):.2f}px")
    else:
        print("[WARN] _meta.npz tidak ada — simulasi piksel (sc=200, center=320,240)")
        seq_unnorm = seq_norm.copy()
        seq_unnorm[:, :, :2] = seq_norm[:, :, :2] * 200 + np.array([320, 240])

    seq_norm_recalc, _, _ = normalize_skeleton(seq_unnorm)

    print(f"File  : {npy_path.name}")
    print(f"Shape : {seq_norm.shape}  (T={seq_norm.shape[0]}, V=17, C=3)")

    if args.frames:
        frames = [int(x) for x in args.frames.split(",")]
    else:
        frames = auto_select_frames(seq_unnorm, seq_norm_recalc, n=args.n)
        print(f"Auto-pilih frame: {frames}")

    T = seq_norm.shape[0]
    frames = [f for f in frames if 0 <= f < T]
    for f in frames:
        out = outdir / f"frame_{f:02d}.png"
        plot_one_frame(seq_unnorm, f, out)

    print("\nSelesai. Gambar tersimpan di:", outdir)


if __name__ == "__main__":
    main()
