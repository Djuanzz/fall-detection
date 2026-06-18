"""
fig_paper_curves.py
===================
Kurva training untuk paper (BACKGROUND PUTIH, standar publikasi).

Output (docs/figures_paper/):
    fig_kurva_training.png          ← GAMBAR_KURVA (Train Loss & Val Accuracy, 4 konfigurasi)
    fig_kurva_training_valloss.png  ← versi Val Loss (skala log) & Val Accuracy, 4 konfigurasi
    fig_kurva_17bal.png             ← per-konfigurasi (Train Loss | Accuracy)
    fig_kurva_17ful.png
    fig_kurva_25bal.png
    fig_kurva_25ful.png
    fig_kurva_17bal_valloss.png     ← per-konfigurasi (Train+Val Loss skala log | Accuracy)
    fig_kurva_17ful_valloss.png
    fig_kurva_25bal_valloss.png
    fig_kurva_25ful_valloss.png

Jalankan:
    conda activate block-gcn
    python scripts/fig_paper_curves.py
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent.parent

# ── Paper style: putih, serif, kontras tinggi ───────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "savefig.facecolor": "white",
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "axes.edgecolor":   "black",
    "axes.linewidth":   0.8,
    "text.color":       "black",
    "axes.labelcolor":  "black",
    "xtick.color":      "black",
    "ytick.color":      "black",
    "grid.color":       "#cccccc",
    "grid.linewidth":   0.5,
})

EXPERIMENTS = [
    {"key": "17bal", "label": "17-sendi Balanced",   "log": "weights_new/04_17bal/log.txt", "color": "#1f77b4", "ls": "-"},
    {"key": "17ful", "label": "17-sendi Imbalanced", "log": "weights_new/04_17ful/log.txt", "color": "#d62728", "ls": "-"},
    {"key": "25bal", "label": "25-sendi Balanced",   "log": "weights_new/04_25bal/log.txt", "color": "#2ca02c", "ls": "--"},
    {"key": "25ful", "label": "25-sendi Imbalanced", "log": "weights_new/04_25ful/log.txt", "color": "#ff7f0e", "ls": "--"},
]


def parse_log(log_path: str) -> dict:
    path = ROOT / log_path
    if not path.exists():
        print(f"[SKIP] {path}")
        return {}

    data = {k: [] for k in ["epoch", "train_loss", "train_acc",
                            "val_loss", "val_bal_acc", "val_sens", "val_spec", "val_auc"]}
    cur_epoch = train_loss = train_acc = None
    val_loss = bal_acc = sens = spec = None

    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"Training epoch:\s*(\d+)", line)
            if m:
                cur_epoch = int(m.group(1))
                train_loss = train_acc = val_loss = bal_acc = sens = spec = None
                continue
            m = re.search(r"Mean training loss:\s*(\d+\.\d+)\.?\s.*Mean training acc:\s*(\d+\.\d+)%", line)
            if m:
                train_loss, train_acc = float(m.group(1)), float(m.group(2)); continue
            m = re.search(r"Mean test loss.*?:\s*(\d+\.\d+)", line)
            if m and cur_epoch is not None:
                val_loss = float(m.group(1)); continue
            m = re.search(r"Balanced Accuracy:\s*(\d+\.\d+)%", line)
            if m and cur_epoch is not None:
                bal_acc = float(m.group(1)); continue
            m = re.search(r"Sensitivity.*?:\s*(\d+\.\d+)%", line)
            if m and cur_epoch is not None:
                sens = float(m.group(1)); continue
            m = re.search(r"Specificity.*?:\s*(\d+\.\d+)%", line)
            if m and cur_epoch is not None:
                spec = float(m.group(1)); continue
            m = re.search(r"AUC-ROC:\s*([\d.]+)", line)
            if m and cur_epoch is not None and bal_acc is not None:
                data["epoch"].append(cur_epoch)
                data["train_loss"].append(train_loss if train_loss is not None else float("nan"))
                data["train_acc"].append(train_acc if train_acc is not None else float("nan"))
                data["val_loss"].append(val_loss if val_loss is not None else float("nan"))
                data["val_bal_acc"].append(bal_acc)
                data["val_sens"].append(sens if sens is not None else float("nan"))
                data["val_spec"].append(spec if spec is not None else float("nan"))
                data["val_auc"].append(float(m.group(1)))
                cur_epoch = train_loss = train_acc = val_loss = bal_acc = sens = spec = None
    return {k: np.array(v) for k, v in data.items()}


def grid(ax):
    ax.grid(True, linestyle="--", alpha=0.6)
    for s in ax.spines.values():
        s.set_edgecolor("black")


# ── Combined (GAMBAR_KURVA) ──────────────────────────────────────────────────────
def plot_combined(outdir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for exp in EXPERIMENTS:
        d = parse_log(exp["log"])
        if not d or len(d["epoch"]) == 0:
            continue
        axes[0].plot(d["epoch"], d["train_loss"], color=exp["color"], ls=exp["ls"], lw=1.6, label=exp["label"])
        axes[1].plot(d["epoch"], d["val_bal_acc"], color=exp["color"], ls=exp["ls"], lw=1.6, label=exp["label"])

    axes[0].set_title("(a) Train Loss terhadap Epoch")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training Loss")
    axes[1].set_title("(b) Val Accuracy terhadap Epoch")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(40, 101)
    for ax in axes:
        grid(ax); ax.legend(fontsize=8, framealpha=0.95, edgecolor="#888")

    plt.tight_layout()
    out = outdir / "fig_kurva_training.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    print(f"  -> {out}")
    plt.close()


# ── Combined VAL LOSS (skala log) ────────────────────────────────────────────────
def plot_combined_valloss(outdir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for exp in EXPERIMENTS:
        d = parse_log(exp["log"])
        if not d or len(d["epoch"]) == 0:
            continue
        axes[0].plot(d["epoch"], d["val_loss"], color=exp["color"], ls=exp["ls"], lw=1.6, label=exp["label"])
        axes[1].plot(d["epoch"], d["val_bal_acc"], color=exp["color"], ls=exp["ls"], lw=1.6, label=exp["label"])

    axes[0].set_title("(a) Val Loss terhadap Epoch (skala log)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Validation Loss")
    axes[0].set_yscale("log")
    axes[1].set_title("(b) Val Accuracy terhadap Epoch")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(40, 101)
    for ax in axes:
        grid(ax); ax.legend(fontsize=8, framealpha=0.95, edgecolor="#888")

    plt.tight_layout()
    out = outdir / "fig_kurva_training_valloss.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    print(f"  -> {out}")
    plt.close()


# ── Per-config ───────────────────────────────────────────────────────────────────
def plot_single(exp, outdir):
    d = parse_log(exp["log"])
    if not d or len(d["epoch"]) == 0:
        return
    ep = d["epoch"]; best_i = int(d["val_bal_acc"].argmax())
    print(f"  {exp['label']}: {len(ep)} ep, best bal_acc={d['val_bal_acc'][best_i]:.2f}% @ ep{ep[best_i]}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(ep, d["train_loss"], color="#1f77b4", lw=1.6, label="Train loss")
    ax.set_title("(a) Train Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); grid(ax)
    ax.legend(fontsize=9, edgecolor="#888")

    ax = axes[1]
    ax.plot(ep, d["train_acc"], color="#1f77b4", lw=1.6, label="Train acc")
    ax.plot(ep, d["val_bal_acc"], color="#2ca02c", lw=1.6, label="Val acc")
    ax.axvline(ep[best_i], color="#ff7f0e", lw=1.2, ls="--",
               label=f"Best ep{ep[best_i]}: {d['val_bal_acc'][best_i]:.2f}%")
    ax.set_title("(b) Akurasi"); ax.set_xlabel("Epoch"); ax.set_ylabel("%")
    ax.set_ylim(40, 101); grid(ax); ax.legend(fontsize=9, edgecolor="#888")

    plt.tight_layout()
    out = outdir / f"fig_kurva_{exp['key']}.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    print(f"  -> {out}")
    plt.close()


# ── Per-config VAL LOSS (train+val, skala log) ───────────────────────────────────
def plot_single_valloss(exp, outdir):
    d = parse_log(exp["log"])
    if not d or len(d["epoch"]) == 0:
        return
    ep = d["epoch"]; best_i = int(d["val_bal_acc"].argmax())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(ep, d["train_loss"], color="#1f77b4", lw=1.6, label="Train loss")
    ax.plot(ep, d["val_loss"], color="#d62728", lw=1.6, label="Val loss")
    ax.set_yscale("log")
    ax.set_title("(a) Loss (skala log)"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); grid(ax)
    ax.legend(fontsize=9, edgecolor="#888")

    ax = axes[1]
    ax.plot(ep, d["train_acc"], color="#1f77b4", lw=1.6, label="Train acc")
    ax.plot(ep, d["val_bal_acc"], color="#2ca02c", lw=1.6, label="Val acc")
    ax.axvline(ep[best_i], color="#ff7f0e", lw=1.2, ls="--",
               label=f"Best ep{ep[best_i]}: {d['val_bal_acc'][best_i]:.2f}%")
    ax.set_title("(b) Akurasi"); ax.set_xlabel("Epoch"); ax.set_ylabel("%")
    ax.set_ylim(40, 101); grid(ax); ax.legend(fontsize=9, edgecolor="#888")

    plt.tight_layout()
    out = outdir / f"fig_kurva_{exp['key']}_valloss.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    print(f"  -> {out}")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="docs/figures_paper")
    args = ap.parse_args()
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    print("=== Kurva gabungan train loss (GAMBAR_KURVA) ===")
    plot_combined(outdir)
    print("=== Kurva gabungan val loss (skala log) ===")
    plot_combined_valloss(outdir)
    print("=== Kurva per-konfigurasi (train loss) ===")
    for exp in EXPERIMENTS:
        plot_single(exp, outdir)
    print("=== Kurva per-konfigurasi (val loss, skala log) ===")
    for exp in EXPERIMENTS:
        plot_single_valloss(exp, outdir)
    print("\nSelesai.")


if __name__ == "__main__":
    main()
