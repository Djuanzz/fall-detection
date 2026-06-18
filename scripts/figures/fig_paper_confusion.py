"""
fig_paper_confusion.py
======================
Confusion matrix untuk paper (BACKGROUND PUTIH, standar publikasi).
Tulisan konfigurasi memakai "Balanced" / "Imbalanced" (BUKAN "Full").

Angka diambil dari baris re-eval terakhir tiap log.txt di weights_new/<set>_*
(test set: 889 not_fall + 218 fall). Set 03 = hasil akhir/skripsi,
01 & 02 = eksperimen pembanding.

Output (docs/figures_paper/):
    set 03 (default, tanpa prefix):
        confusion_17bal.png / 17ful / 25bal / 25ful
        confusion_17.png  (17 Bal | Imbal)   confusion_25.png  (25 Bal | Imbal)
        confusion_all.png (2x2 gabungan)
    set 01 & 02 (prefix): confusion_01_*.png, confusion_02_*.png

Jalankan:
    conda activate block-gcn
    python scripts/fig_paper_confusion.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent.parent
OUT_DIR = ROOT / "docs" / "figures_paper"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Paper style: putih, serif, kontras tinggi ───────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "axes.edgecolor":    "black",
    "axes.linewidth":    0.8,
    "text.color":        "black",
    "axes.labelcolor":   "black",
    "xtick.color":       "black",
    "ytick.color":       "black",
})

# cm = [[TN, FP], [FN, TP]]  (baris = aktual, kolom = prediksi)
LABELS = {
    "17bal": "17-sendi Balanced",
    "17ful": "17-sendi Imbalanced",
    "25bal": "25-sendi Balanced",
    "25ful": "25-sendi Imbalanced",
}

# Setiap set: cm tiap konfig diambil dari baris "TP= .. TN= .. FP= .. FN= .."
# pada EPOCH CHECKPOINT (best balanced accuracy), BUKAN epoch terakhir.
# Epoch checkpoint = angka di nama file runs-<epoch>-*.pt / baris "Best Bal Acc (epoch N)".
SETS = {
    "04": {  # checkpoint: 17bal ep61, 17ful ep38, 25bal ep52, 25ful ep76
        "17bal": [[873, 16], [3, 215]],
        "17ful": [[887, 2], [3, 215]],
        "25bal": [[848, 41], [5, 213]],
        "25ful": [[881, 8], [5, 213]],
    },
    "03": {
        "17bal": [[884, 5], [2, 216]],
        "17ful": [[889, 0], [2, 216]],
        "25bal": [[875, 14], [2, 216]],
        "25ful": [[877, 12], [1, 217]],
    },
    "02": {
        "17bal": [[874, 15], [2, 216]],
        "17ful": [[830, 59], [3, 215]],
        "25bal": [[875, 14], [4, 214]],
        "25ful": [[877, 12], [2, 216]],
    },
    "01": {
        "17bal": [[874, 15], [2, 216]],
        "17ful": [[883, 6], [4, 214]],
        "25bal": [[876, 13], [4, 214]],
        "25ful": [[881, 8], [2, 216]],
    },
}


def experiments(set_name):
    """Confusion matrix per konfig.

    Prioritas: JSON hasil re-eval offline checkpoint
    (docs/eval/confusion_<set>.json dari scripts/eval_all_scenarios.py).
    Kalau JSON belum ada, fallback ke angka hardcoded di SETS.
    """
    import json
    json_path = ROOT / "docs" / "eval" / f"confusion_{set_name}.json"
    if json_path.exists():
        data = json.loads(json_path.read_text())
        print(f"[{set_name}] pakai re-eval offline: {json_path}")
        return [{"key": k, "label": LABELS[k],
                 "cm": [[data[k]["TN"], data[k]["FP"]],
                        [data[k]["FN"], data[k]["TP"]]]}
                for k in LABELS if k in data]
    print(f"[{set_name}] JSON belum ada -> pakai angka hardcoded (fallback)")
    return [{"key": k, "label": LABELS[k], "cm": cm}
            for k, cm in SETS[set_name].items()]

CLASSES = ["Tidak Jatuh", "Jatuh"]  # not_fall, fall


def draw_cm(ax, cm, title):
    cm = np.array(cm)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Prediksi\n" + c for c in CLASSES])
    ax.set_yticklabels(["Aktual\n" + c for c in CLASSES], rotation=90, va="center")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    thr = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black",
                    fontsize=20, fontweight="bold")
    ax.set_title(title, pad=10)
    # garis grid tipis antar sel
    ax.set_xticks([0.5], minor=True)
    ax.set_yticks([0.5], minor=True)
    ax.grid(which="minor", color="black", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    return im


def generate(set_name, prefix):
    exps = experiments(set_name)
    by_key = {e["key"]: e for e in exps}

    # ── Per-konfigurasi ─────────────────────────────────────────────────────────
    for exp in exps:
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        im = draw_cm(ax, exp["cm"], "Confusion Matrix\n" + exp["label"])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out = OUT_DIR / f"confusion_{prefix}{exp['key']}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out}")

    # ── Per-jumlah sendi (2 panel: Balanced | Imbalanced) ───────────────────────
    for joints, keys in {"17": ["17bal", "17ful"], "25": ["25bal", "25ful"]}.items():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
        for ax, k in zip(axes.ravel(), keys):
            draw_cm(ax, by_key[k]["cm"], by_key[k]["label"])
        fig.subplots_adjust(wspace=0.45, left=0.08, right=0.97, top=0.90, bottom=0.12)
        out = OUT_DIR / f"confusion_{prefix}{joints}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {out}")

    # ── Gabungan 2x2 (tanpa judul atas, jarak antar panel dilebihin) ────────────
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 10))
    for ax, exp in zip(axes.ravel(), exps):
        draw_cm(ax, exp["cm"], exp["label"])
    fig.subplots_adjust(wspace=0.45, hspace=0.45,
                        left=0.07, right=0.97, top=0.95, bottom=0.07)
    out = OUT_DIR / f"confusion_{prefix}all.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def main():
    # generate("03", prefix="")      # hasil akhir: tanpa prefix
    generate("04", prefix="04_")
    # generate("02", prefix="02_")
    # generate("01", prefix="01_")


if __name__ == "__main__":
    main()
