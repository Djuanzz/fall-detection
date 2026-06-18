"""
fig_skeleton_high_vs_oklusi.py
==============================
Dua montage TERPISAH ekstraksi skeleton (5 aksi sebaris, 'jatuh' paling kanan),
nama aksi ditulis di bawah tiap gambar. Tanpa label baris di kiri.

Output (paper_shots/skeleton/):
    montage_high.png    -> ekstraksi conf tinggi (berhasil)
    montage_oklusi.png  -> ekstraksi saat oklusi (sebagian sendi merah)

Cara pakai:
    python scripts/fig_skeleton_high_vs_oklusi.py
"""

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent.parent

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "savefig.dpi": 300,
    "figure.dpi": 300,
})

# urutan kolom: jatuh paling kanan
ACTIONS = [
    ("Duduk",        "duduk/tinggi/hi_018_f76_c0.97.png",
                     "duduk_oklusi/oklusi/occ_016_f42_low4_miss0.png"),
    ("Berdiri",      "berdiri/tinggi/hi_013_f56_c0.97.png",
                     "berdiri_oklusi/oklusi/occ_010_f36_low4_miss0.png"),
    ("Lompat",       "lompat/tinggi/hi_006_f28_c0.97.png",
                     "lompat_oklusi/oklusi/occ_017_f38_low4_miss0.png"),
    ("Sempoyongan",  "sempoyongan/tinggi/hi_021_f88_c0.97.png",
                     "sempoyongan_oklusi/oklusi/occ_029_f74_low5_miss0.png"),
    ("Jatuh",        "jatuh/tinggi/hi_013_f56_c0.96.png",
                     "jatuh_oklusi/oklusi/occ_007_f52_low12_miss0.png"),
]

BASE = ROOT / "paper_shots" / "skeleton"


def load_rgb(rel):
    p = BASE / rel
    img = cv2.imread(str(p))
    if img is None:
        raise FileNotFoundError(p)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def make_montage(which, out_path):
    """which = 1 (kolom hi) atau 2 (kolom oklusi) pada tuple ACTIONS."""
    n = len(ACTIONS)
    fig, axes = plt.subplots(
        1, n, figsize=(2.0 * n, 4.2), facecolor="white",
        gridspec_kw={"wspace": 0.04})
    for c, item in enumerate(ACTIONS):
        name, rel = item[0], item[which]
        ax = axes[c]
        ax.imshow(load_rgb(rel))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#bbbbbb"); s.set_linewidth(0.8)
        ax.set_xlabel(name, fontsize=13, fontweight="bold",
                      labelpad=8, color="#111111")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Tersimpan: {out_path}")
    plt.close()


def main():
    make_montage(1, BASE / "montage_high.png")
    make_montage(2, BASE / "montage_oklusi.png")


if __name__ == "__main__":
    main()
