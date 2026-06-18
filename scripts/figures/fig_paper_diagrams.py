"""
fig_paper_diagrams.py
=====================
Diagram untuk paper (BACKGROUND PUTIH, garis hitam, gaya publikasi).

Output (docs/figures_paper/):
    fig_peta_sendi_coco.png       ← GAMBAR_PETA_SENDI_COCO  (17 sendi COCO + nomor)
    fig_pipeline_ekstraksi.png    ← GAMBAR_PIPELINE_EKSTRAKSI (alur ekstraksi skeleton)
    fig_flow_deteksi_video.png    ← GAMBAR_FLOW_DETEKSI (sliding window video)
    fig_arsitektur_realtime.png   ← GAMBAR_ARSITEKTUR_REALTIME (sistem near real-time)

Jalankan:
    python scripts/fig_paper_diagrams.py
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = Path(__file__).parent.parent.parent
OUT = ROOT / "docs/figures_paper"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

EDGE = "#222222"


def box(ax, x, y, w, h, text, fc="white", ec=EDGE, fontsize=9, bold=False, sub=None):
    """Kotak proses dengan teks tengah. (x,y) = pusat."""
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.06",
                       linewidth=1.3, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    if sub:
        ax.text(x, y + h * 0.16, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold" if bold else "normal", color="black")
        ax.text(x, y - h * 0.22, sub, ha="center", va="center",
                fontsize=fontsize - 2.0, color="#444444")
    else:
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold" if bold else "normal", color="black")


def diamond(ax, x, y, w, h, text, fc="#fff6e6", sub=None):
    pts = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
    ax.add_patch(plt.Polygon(pts, closed=True, linewidth=1.3, edgecolor="#cc8800", facecolor=fc))
    ax.text(x, y + (h * 0.13 if sub else 0), text, ha="center", va="center", fontsize=8.5, fontweight="bold")
    if sub:
        ax.text(x, y - h * 0.2, sub, ha="center", va="center", fontsize=7, color="#555")


def arrow(ax, x1, y1, x2, y2, text=None, color=EDGE):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.2, color=color, shrinkA=2, shrinkB=2))
    if text:
        ax.text((x1 + x2) / 2 + 0.12, (y1 + y2) / 2, text, fontsize=7.5, color=color, ha="left", va="center")


# ── 1. Peta sendi COCO 17 ────────────────────────────────────────────────────────
def fig_coco():
    # posisi (x, y) menyerupai tubuh menghadap pengamat; y ke atas
    P = {
        0: (0.0, 9.3), 1: (-0.35, 9.6), 2: (0.35, 9.6), 3: (-0.7, 9.4), 4: (0.7, 9.4),
        5: (-1.1, 8.3), 6: (1.1, 8.3), 7: (-1.7, 6.9), 8: (1.7, 6.9),
        9: (-2.1, 5.6), 10: (2.1, 5.6), 11: (-0.7, 5.7), 12: (0.7, 5.7),
        13: (-0.85, 3.6), 14: (0.85, 3.6), 15: (-0.95, 1.6), 16: (0.95, 1.6),
    }
    # subjek MENGHADAP kamera -> sisi "Right" tubuh tampil di KIRI pengamat.
    # cermin koordinat x agar tata letak benar secara anatomis.
    P = {i: (-x, y) for i, (x, y) in P.items()}

    names = {0: "Nose", 1: "Left Eye", 2: "Right Eye", 3: "Left Ear", 4: "Right Ear",
             5: "Left Shoulder", 6: "Right Shoulder", 7: "Left Elbow", 8: "Right Elbow",
             9: "Left Wrist", 10: "Right Wrist", 11: "Left Hip", 12: "Right Hip",
             13: "Left Knee", 14: "Right Knee", 15: "Left Ankle", 16: "Right Ankle"}
    edges = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
             (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

    # warna per-region (palet Okabe-Ito, colorblind-safe, kontras utk paper)
    HEAD, LARM, RARM, LLEG, RLEG = "#E69F00", "#0072B2", "#D55E00", "#009E73", "#CC79A7"
    COL = {0: HEAD, 1: HEAD, 2: HEAD, 3: HEAD, 4: HEAD,
           5: LARM, 7: LARM, 9: LARM, 6: RARM, 8: RARM, 10: RARM,
           11: LLEG, 13: LLEG, 15: LLEG, 12: RLEG, 14: RLEG, 16: RLEG}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.6, 5.6),
                                   gridspec_kw={"width_ratios": [1, 1], "wspace": 0.0})

    # --- panel kiri: skeleton ---
    for a, b in edges:
        axL.plot([P[a][0], P[b][0]], [P[a][1], P[b][1]], color="#999999", lw=2, zorder=1)
    for i, (x, y) in P.items():
        axL.add_patch(Circle((x, y), 0.19, facecolor=COL[i], edgecolor="black", lw=0.8, zorder=2))
        dx = 0.38 if x >= 0 else -0.38
        ha = "left" if x >= 0 else "right"
        axL.text(x + dx, y, f"{i}", fontsize=11, fontweight="bold", ha=ha, va="center", color="black")
    axL.set_xlim(-2.7, 2.7)            # margin kiri = kanan (simetris)
    axL.set_ylim(0.8, 10.1)
    axL.set_aspect("equal"); axL.axis("off")

    # --- panel kanan: legend dgn marker warna sesuai sendi ---
    n = 17
    top, bot = 0.93, 0.07
    step = (top - bot) / (n - 1)
    for k, i in enumerate(range(n)):
        yy = top - k * step
        axR.scatter(0.06, yy, s=70, c=COL[i], edgecolors="black", linewidths=0.6,
                    transform=axR.transAxes, zorder=2, clip_on=False)
        axR.text(0.13, yy, f"{i}. {names[i]}",
                 fontsize=10.5, va="center", ha="left", color="black",
                 transform=axR.transAxes)
    axR.set_xlim(0, 1); axR.set_ylim(0, 1); axR.axis("off")

    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    out = OUT / "fig_peta_sendi_coco.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight"); plt.close()
    print(f"  -> {out}")


# ── 2. Pipeline ekstraksi ────────────────────────────────────────────────────────
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(6, 9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 15); ax.axis("off")
    cx, w, h = 5, 5.2, 1.05
    steps = [
        (13.6, "Video NTU RGB+D", "5 aksi: A043 (fall) / A008,A009,A027,A042 (not_fall)", "#eef4ff"),
        (11.9, "Cuplik 150 frame seragam", "sampling merata sepanjang durasi", "white"),
        (10.2, "YOLO11n-pose", "17 keypoint COCO + confidence / frame", "#eef4ff"),
        (8.5, "Seleksi subjek", "argmax rata-rata confidence -> 1 orang", "white"),
    ]
    ys = [s[0] for s in steps]
    for (y, t, s, fc) in steps:
        box(ax, cx, y, w, h, t, fc=fc, bold=True, sub=s)
    for i in range(len(ys) - 1):
        arrow(ax, cx, ys[i] - h / 2, cx, ys[i + 1] + h / 2)

    # decision: cukup frame valid?
    arrow(ax, cx, 8.5 - h / 2, cx, 7.35)
    diamond(ax, cx, 6.7, 3.6, 1.5, "> 40% frame valid?", sub="conf>0.2, min 1 joint")
    box(ax, 1.7, 6.7, 2.2, 0.9, "Tolak video", fc="#ffecec", ec="#cc3333", bold=True)
    arrow(ax, cx - 1.8, 6.7, 1.7 + 1.1, 6.7, text="Tidak", color="#cc3333")

    arrow(ax, cx, 6.7 - 0.75, cx, 5.45, text="Ya", color="#2a8a2a")
    box(ax, cx, 4.9, w, h, "Normalisasi", fc="white", bold=True,
        sub="translasi hip-center + skala lebar bahu = 1")
    arrow(ax, cx, 4.9 - h / 2, cx, 3.65)
    box(ax, cx, 3.1, w, h, "Simpan array (T, 17, 3)", fc="#e9f7ef", bold=True,
        sub="kanal: x, y, confidence")
    arrow(ax, cx, 3.1 - h / 2, cx, 1.85)
    box(ax, cx, 1.3, w, h, "prepare_dataset_yolo17.py", fc="#f3e9ff", bold=True,
        sub="cross-subject split, balanced / full")

    ax.set_title("Pipeline Ekstraksi Skeleton YOLO11-pose", fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout()
    out = OUT / "fig_pipeline_ekstraksi.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight"); plt.close()
    print(f"  -> {out}")


# ── 3. Flow deteksi video (sliding window) ───────────────────────────────────────
def fig_flow_video():
    fig, ax = plt.subplots(figsize=(6, 9.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 15.5); ax.axis("off")
    cx, w, h = 5, 5.4, 1.0
    steps = [
        (14.3, "Baca frame video", "frame demi frame", "#eef4ff"),
        (12.7, "YOLO11-pose -> 17 keypoint", "ekstraksi pose per frame", "white"),
        (11.1, "Smoothing EMA (alpha=0,4)", "redam getaran posisi sendi", "white"),
        (9.5, "Buffer sliding window 64 frame", "antrian keypoint terbaru", "#eef4ff"),
    ]
    ys = [s[0] for s in steps]
    for (y, t, s, fc) in steps:
        box(ax, cx, y, w, h, t, fc=fc, bold=True, sub=s)
    for i in range(len(ys) - 1):
        arrow(ax, cx, ys[i] - h / 2, cx, ys[i + 1] + h / 2)

    arrow(ax, cx, 9.5 - h / 2, cx, 8.35)
    diamond(ax, cx, 7.7, 4.0, 1.5, "buffer penuh &", sub="step 15 frame?")
    box(ax, 1.6, 7.7, 2.3, 0.9, "Tunggu frame", fc="#f5f5f5", bold=True)
    arrow(ax, cx - 2.0, 7.7, 1.6 + 1.15, 7.7, text="Tidak")
    # loop balik
    arrow(ax, 1.6, 7.7 + 0.45, 1.6, 14.3, color="#888")
    arrow(ax, 1.6, 14.3, cx - w / 2, 14.3, color="#888")

    arrow(ax, cx, 7.7 - 0.75, cx, 6.5, text="Ya", color="#2a8a2a")
    box(ax, cx, 5.95, w, h, "Inferensi BlockGCN (17-sendi Full)", fc="#eef4ff", bold=True,
        sub="probabilitas kelas fall / not_fall")
    arrow(ax, cx, 5.95 - h / 2, cx, 4.75)
    box(ax, cx, 4.2, w, h, "Voting 5 window terakhir", fc="white", bold=True,
        sub="rata-rata probabilitas")
    arrow(ax, cx, 4.2 - h / 2, cx, 3.0)
    diamond(ax, cx, 2.3, 3.8, 1.5, "P(fall) >= 0,5 ?")
    box(ax, 1.9, 2.3, 2.0, 0.95, "Status:\nFALL", fc="#ffecec", ec="#cc3333", bold=True)
    box(ax, 8.1, 2.3, 2.0, 0.95, "Status:\nNORMAL", fc="#e9f7ef", ec="#2a8a2a", bold=True)
    arrow(ax, cx - 1.9, 2.3, 1.9 + 1.0, 2.3, text="Ya", color="#cc3333")
    arrow(ax, cx + 1.9, 2.3, 8.1 - 1.0, 2.3, text="Tidak", color="#2a8a2a")

    ax.set_title("Alur Deteksi Jatuh pada Video", fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout()
    out = OUT / "fig_flow_deteksi_video.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight"); plt.close()
    print(f"  -> {out}")


# ── 4. Arsitektur near real-time ─────────────────────────────────────────────────
def fig_arsitektur():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 22); ax.set_ylim(0, 9); ax.axis("off")
    yc, w, h = 5.5, 3.2, 1.5
    xs = [2.3, 6.2, 10.1, 14.0, 17.9]
    blocks = [
        ("Kamera", "iPhone via Camo\n1280x720", "#eef4ff"),
        ("YOLO11m-pose", "input 640\n17 keypoint", "white"),
        ("Praproses", "EMA alpha=0,4\nnormalisasi", "white"),
        ("BlockGCN", "17-sendi Full\nP(fall)", "#eef4ff"),
        ("Keputusan", "voting 5 window\n+ gating", "white"),
    ]
    for x, (t, s, fc) in zip(xs, blocks):
        box(ax, x, yc, w, h, t, fc=fc, bold=True, sub=s.replace("\n", "  "))
    for i in range(len(xs) - 1):
        arrow(ax, xs[i] + w / 2, yc, xs[i + 1] - w / 2, yc)

    # output panel
    box(ax, 17.9, 2.4, w, 1.4, "Tampilan:\nskeleton + status + P(fall)", fc="#fff6e6", ec="#cc8800")
    arrow(ax, 17.9, yc - h / 2, 17.9, 2.4 + 0.7)

    # gating note
    box(ax, 10.1, 2.4, 6.0, 1.3, "Gating: inferensi dilewati bila frame valid terlalu sedikit",
        fc="#f5f5f5", ec="#888", fontsize=8.5)
    arrow(ax, 10.1, yc - h / 2, 10.1, 2.4 + 0.65, color="#888")

    ax.set_title("Arsitektur Sistem Deteksi Jatuh Near Real-Time", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()
    out = OUT / "fig_arsitektur_realtime.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight"); plt.close()
    print(f"  -> {out}")


def main():
    print("=== Diagram paper (white bg) ===")
    fig_coco()
    fig_pipeline()
    fig_flow_video()
    fig_arsitektur()
    print("Selesai.")


if __name__ == "__main__":
    main()
