"""
eda_skeleton_dataset.py
=======================
EDA (Exploratory Data Analysis) untuk skeleton MURNI hasil ekstraksi
YOLO11n-pose (sebelum padding/cropping di prepare_dataset).

Input folder:
    dataset/ntu_skeleton/
        fall/      <- *.npy shape (T, 17, 3)  [x, y, conf]   T bervariasi!
        not_fall/  <- *.npy shape (T, 17, 3)
        file_index.json
        label_map.json
        extraction_stats.json

Fokus utama: distribusi jumlah frame T per video (hasil subsample
np.linspace(0, total, min(total, max_frames=150)) di extract_skeleton.py).

Plus statistik:
  - Jumlah file per class (fall vs not_fall)
  - Distribusi T per video (histogram + boxplot per class)
  - Per-action-class breakdown (A008, A009, A027, A042, A043)
  - Persentase frame valid per video (frame dengan minimal 1 joint deteksi)
  - Distribusi confidence rata-rata per video
  - Frekuensi deteksi per joint (joint mana sering missing di video)
  - Heatmap rate deteksi per joint vs posisi-relatif-frame

Cara pakai:
    python scripts/eda_skeleton_dataset.py
    python scripts/eda_skeleton_dataset.py --skeleton_dir dataset/ntu_skeleton
    python scripts/eda_skeleton_dataset.py --out_dir docs-pipeline-fix/eda
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install matplotlib")

# COCO 17-joint names
JOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

CLASS_NAMES = {0: "not_fall", 1: "fall"}
CLASS_COLORS = {0: "#2ecc71", 1: "#e74c3c"}   # green, red

ACTION_NAMES = {
    8:  "sit_down",
    9:  "stand_up",
    27: "jump_up",
    42: "stagger",
    43: "fall_down",
}

ACTION_COLORS = {
    8:  "#1abc9c",
    9:  "#3498db",
    27: "#9b59b6",
    42: "#f39c12",
    43: "#e74c3c",
}


def parse_action_class(stem: str):
    """Extract action ID dari nama file NTU: S001C001P001R001A043 → 43."""
    m = re.search(r'[Aa](\d{3})', stem)
    return int(m.group(1)) if m else None


def parse_subject(stem: str):
    m = re.search(r'[Pp](\d{3})', stem)
    return int(m.group(1)) if m else None


# ── Loader ─────────────────────────────────────────────────────────────────────

def scan_skeleton_dir(skeleton_dir: Path):
    """
    Scan fall/ dan not_fall/ subfolder. Return list of dict per file:
        {
            "path", "stem", "label" (0/1), "action" (int),
            "T", "valid_T", "mean_conf", "joint_rate" (V,), "heat" (T,V),
        }
    """
    records = []
    for label_name, label_idx in (("not_fall", 0), ("fall", 1)):
        sub = skeleton_dir / label_name
        if not sub.exists():
            print(f"  [WARN] Folder tidak ada: {sub}")
            continue
        files = sorted(sub.glob("*.npy"))
        print(f"  Scan {label_name:10s} (label={label_idx}): {len(files)} file")
        for fp in files:
            try:
                arr = np.load(str(fp), allow_pickle=False)
            except Exception as e:
                print(f"    [SKIP] {fp.name}: {e}")
                continue

            if arr.ndim != 3 or arr.shape[1] != 17 or arr.shape[2] < 2:
                print(f"    [SKIP] shape aneh {arr.shape}: {fp.name}")
                continue

            # Pastikan 3 channel
            if arr.shape[2] == 2:
                conf = np.ones((arr.shape[0], arr.shape[1], 1), dtype=np.float32)
                arr = np.concatenate([arr, conf], axis=2)

            T = arr.shape[0]
            conf2d = arr[:, :, 2]                         # (T, V)
            valid_mask = (conf2d > 0).any(axis=1)
            valid_T = int(valid_mask.sum())

            mask_conf = conf2d > 0
            mean_conf = float(conf2d[mask_conf].mean()) if mask_conf.any() else 0.0

            joint_rate = (conf2d > 0).astype(np.float32).mean(axis=0)  # (V,)
            heat = (conf2d > 0).astype(np.float32)                      # (T, V)

            records.append({
                "stem":       fp.stem,
                "label":      label_idx,
                "action":     parse_action_class(fp.stem),
                "subject":    parse_subject(fp.stem),
                "T":          T,
                "valid_T":    valid_T,
                "mean_conf":  mean_conf,
                "joint_rate": joint_rate,
                "heat":       heat,
            })
    return records


# ── Stats printer ──────────────────────────────────────────────────────────────

def print_global_stats(records):
    print(f"\n{'='*70}")
    print(f"  STATISTIK GLOBAL")
    print(f"{'='*70}")

    total = len(records)
    n_fall = sum(1 for r in records if r["label"] == 1)
    n_not  = total - n_fall

    print(f"  Total file       : {total}")
    print(f"  fall (label 1)   : {n_fall}")
    print(f"  not_fall (lbl 0) : {n_not}")
    if n_fall:
        print(f"  Ratio (nf : f)   : {n_not/n_fall:.2f} : 1")

    Ts       = np.array([r["T"]       for r in records])
    valid_Ts = np.array([r["valid_T"] for r in records])
    confs    = np.array([r["mean_conf"] for r in records])

    print(f"\n  Distribusi panjang sequence T (raw):")
    print(f"    min={int(Ts.min())}  max={int(Ts.max())}  "
          f"mean={Ts.mean():.1f}  median={np.median(Ts):.1f}  std={Ts.std():.1f}")
    print(f"    p25={np.percentile(Ts,25):.0f}  p75={np.percentile(Ts,75):.0f}  "
          f"p95={np.percentile(Ts,95):.0f}")

    print(f"\n  Distribusi frame valid (≥1 joint terdeteksi):")
    print(f"    min={int(valid_Ts.min())}  max={int(valid_Ts.max())}  "
          f"mean={valid_Ts.mean():.1f}  median={np.median(valid_Ts):.1f}")
    pct_valid = valid_Ts / np.maximum(Ts, 1) * 100
    print(f"    %valid (rata-rata): {pct_valid.mean():.1f}%")

    print(f"\n  Confidence rata-rata per file:")
    print(f"    min={confs.min():.3f}  max={confs.max():.3f}  mean={confs.mean():.3f}")


def print_per_class_stats(records):
    print(f"\n{'─'*70}")
    print(f"  STATISTIK PER LABEL")
    print(f"{'─'*70}")
    for lbl in (0, 1):
        sub = [r for r in records if r["label"] == lbl]
        if not sub:
            continue
        Ts = np.array([r["T"] for r in sub])
        vTs = np.array([r["valid_T"] for r in sub])
        confs = np.array([r["mean_conf"] for r in sub])
        print(f"\n  [{CLASS_NAMES[lbl]}]  n={len(sub)}")
        print(f"    T raw     : min={int(Ts.min()):3d}  max={int(Ts.max()):3d}  "
              f"mean={Ts.mean():.1f}  median={np.median(Ts):.1f}")
        print(f"    T valid   : min={int(vTs.min()):3d}  max={int(vTs.max()):3d}  "
              f"mean={vTs.mean():.1f}  median={np.median(vTs):.1f}")
        print(f"    Mean conf : {confs.mean():.3f}")


def print_per_action_stats(records):
    print(f"\n{'─'*70}")
    print(f"  STATISTIK PER ACTION CLASS NTU")
    print(f"{'─'*70}")
    groups = defaultdict(list)
    for r in records:
        if r["action"] is not None:
            groups[r["action"]].append(r)
    for cls in sorted(groups.keys()):
        sub = groups[cls]
        Ts = np.array([r["T"] for r in sub])
        vTs = np.array([r["valid_T"] for r in sub])
        confs = np.array([r["mean_conf"] for r in sub])
        name = ACTION_NAMES.get(cls, "unknown")
        print(f"  A{cls:03d} {name:12s} n={len(sub):4d}  "
              f"T mean={Ts.mean():5.1f}  T_valid mean={vTs.mean():5.1f}  "
              f"conf={confs.mean():.3f}")


def print_subject_distribution(records):
    print(f"\n{'─'*70}")
    print(f"  DISTRIBUSI PER SUBJECT (P-ID NTU)")
    print(f"{'─'*70}")
    subs = Counter(r["subject"] for r in records if r["subject"] is not None)
    print(f"  Total subjek unik: {len(subs)}")
    for sid in sorted(subs.keys()):
        print(f"    P{sid:03d}: {subs[sid]} file")


def print_joint_rate(records):
    print(f"\n{'─'*70}")
    print(f"  RATE DETEKSI PER JOINT (rata-rata global)")
    print(f"{'─'*70}")
    all_rates = np.stack([r["joint_rate"] for r in records])  # (N, V)
    mean_rate = all_rates.mean(axis=0)
    order = np.argsort(mean_rate)
    for i in order:
        bar = "█" * int(mean_rate[i] * 30)
        print(f"    {JOINT_NAMES[i]:18s} {mean_rate[i]*100:5.1f}%  {bar}")
    return mean_rate


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_T_distribution(records, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    Ts = np.array([r["T"] for r in records])
    max_T = int(Ts.max())
    bins = np.linspace(0, max_T + 5, 31)

    # Histogram per class
    ax = axes[0]
    for lbl in (0, 1):
        sub = [r["T"] for r in records if r["label"] == lbl]
        if sub:
            ax.hist(sub, bins=bins, alpha=0.6, color=CLASS_COLORS[lbl],
                    label=f"{CLASS_NAMES[lbl]} (n={len(sub)}, "
                          f"mean={np.mean(sub):.0f})",
                    edgecolor="black")
    ax.set_xlabel("Jumlah frame T per video")
    ax.set_ylabel("Frekuensi")
    ax.set_title("Distribusi T (raw frame count, hasil subsample 150)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Boxplot per class
    ax = axes[1]
    data_box = [
        [r["T"] for r in records if r["label"] == lbl]
        for lbl in (0, 1)
    ]
    lbl_box = [CLASS_NAMES[lbl] for lbl in (0, 1)]
    bp = ax.boxplot(data_box, labels=lbl_box, patch_artist=True, showmeans=True)
    for patch, lbl in zip(bp["boxes"], (0, 1)):
        patch.set_facecolor(CLASS_COLORS[lbl])
        patch.set_alpha(0.6)
    ax.set_ylabel("T per video")
    ax.set_title("Boxplot T per class")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot tersimpan: {out_path}")


def plot_valid_T_distribution(records, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    max_T = max(r["valid_T"] for r in records)
    bins = np.linspace(0, max_T + 5, 31)

    # Histogram per class
    ax = axes[0]
    for lbl in (0, 1):
        sub = [r["valid_T"] for r in records if r["label"] == lbl]
        if sub:
            ax.hist(sub, bins=bins, alpha=0.6, color=CLASS_COLORS[lbl],
                    label=f"{CLASS_NAMES[lbl]} (mean={np.mean(sub):.0f})",
                    edgecolor="black")
    ax.set_xlabel("Frame valid per video (≥1 joint terdeteksi)")
    ax.set_ylabel("Frekuensi")
    ax.set_title("Distribusi T_valid per class")
    ax.legend()
    ax.grid(alpha=0.3)

    # Boxplot
    ax = axes[1]
    data_box = [
        [r["valid_T"] for r in records if r["label"] == lbl]
        for lbl in (0, 1)
    ]
    lbl_box = [CLASS_NAMES[lbl] for lbl in (0, 1)]
    bp = ax.boxplot(data_box, labels=lbl_box, patch_artist=True, showmeans=True)
    for patch, lbl in zip(bp["boxes"], (0, 1)):
        patch.set_facecolor(CLASS_COLORS[lbl])
        patch.set_alpha(0.6)
    ax.set_ylabel("T_valid")
    ax.set_title("Boxplot T_valid per class")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot tersimpan: {out_path}")


def plot_T_per_action(records, out_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    groups = defaultdict(list)
    for r in records:
        if r["action"] is not None:
            groups[r["action"]].append(r["T"])

    keys = sorted(groups.keys())
    data = [groups[k] for k in keys]
    labels_x = [f"A{k:03d}\n{ACTION_NAMES.get(k,'?')}\nn={len(groups[k])}" for k in keys]
    colors = [ACTION_COLORS.get(k, "#7f8c8d") for k in keys]

    bp = ax.boxplot(data, labels=labels_x, patch_artist=True, showmeans=True)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
    ax.set_ylabel("T per video")
    ax.set_title("Boxplot T per NTU action class")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot tersimpan: {out_path}")


def plot_pct_valid(records, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 100, 31)
    for lbl in (0, 1):
        sub = [
            r["valid_T"] / max(r["T"], 1) * 100
            for r in records if r["label"] == lbl
        ]
        if sub:
            ax.hist(sub, bins=bins, alpha=0.6, color=CLASS_COLORS[lbl],
                    label=f"{CLASS_NAMES[lbl]} (mean={np.mean(sub):.1f}%)",
                    edgecolor="black")
    ax.set_xlabel("% frame valid per video")
    ax.set_ylabel("Frekuensi")
    ax.set_title("Distribusi % frame valid (T_valid / T)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot tersimpan: {out_path}")


def plot_confidence(records, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 41)
    for lbl in (0, 1):
        sub = [r["mean_conf"] for r in records if r["label"] == lbl]
        if sub:
            ax.hist(sub, bins=bins, alpha=0.6, color=CLASS_COLORS[lbl],
                    label=f"{CLASS_NAMES[lbl]} (mean={np.mean(sub):.3f})",
                    edgecolor="black")
    ax.set_xlabel("Mean confidence per video")
    ax.set_ylabel("Frekuensi")
    ax.set_title("Distribusi mean confidence")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot tersimpan: {out_path}")


def plot_joint_rate(mean_rate, out_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    idx = np.arange(len(mean_rate))
    bars = ax.bar(idx, mean_rate * 100, color="#3498db", edgecolor="black")
    for b, r in zip(bars, mean_rate):
        ax.text(b.get_x() + b.get_width()/2, r*100 + 0.5,
                f"{r*100:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(idx)
    ax.set_xticklabels(JOINT_NAMES, rotation=45, ha="right")
    ax.set_ylabel("Detection rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Rate deteksi per joint (rata-rata global)")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot tersimpan: {out_path}")


def plot_heatmap_relative(records, out_path, n_bins=30):
    """
    Heatmap (relative_frame_position [0..1] × joint).
    Karena T bervariasi, posisi frame dinormalisasi ke 0-1.
    """
    V = 17
    heat = np.zeros((n_bins, V), dtype=np.float64)
    count = np.zeros((n_bins, V), dtype=np.float64)

    for r in records:
        h = r["heat"]   # (T, V)
        T = h.shape[0]
        for t in range(T):
            rel = t / max(T - 1, 1)
            bi = min(int(rel * n_bins), n_bins - 1)
            heat[bi] += h[t]
            count[bi] += 1.0

    rate = heat / np.maximum(count, 1.0)

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(rate.T, aspect="auto", cmap="viridis",
                   origin="lower", vmin=0, vmax=1,
                   extent=[0, 1, -0.5, V - 0.5])
    ax.set_xlabel("Posisi frame relatif (0=awal, 1=akhir)")
    ax.set_ylabel("Joint")
    ax.set_yticks(np.arange(V))
    ax.set_yticklabels(JOINT_NAMES, fontsize=8)
    ax.set_title(f"Heatmap rate deteksi (posisi-relatif × joint)  bins={n_bins}")
    plt.colorbar(im, ax=ax, label="Detection rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot tersimpan: {out_path}")


def plot_T_scatter(records, out_path):
    """Scatter: T raw vs T valid, warna per class."""
    fig, ax = plt.subplots(figsize=(8, 8))
    for lbl in (0, 1):
        sub = [(r["T"], r["valid_T"]) for r in records if r["label"] == lbl]
        if not sub:
            continue
        xs, ys = zip(*sub)
        ax.scatter(xs, ys, s=10, alpha=0.4, color=CLASS_COLORS[lbl],
                   label=f"{CLASS_NAMES[lbl]} (n={len(sub)})")
    lim = max(max(r["T"] for r in records), max(r["valid_T"] for r in records)) + 5
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, label="y=x (semua valid)")
    ax.set_xlabel("T raw (jumlah frame setelah subsample)")
    ax.set_ylabel("T valid (frame dengan deteksi)")
    ax.set_title("T raw vs T valid per video")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", "box")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot tersimpan: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="EDA skeleton murni di dataset/ntu_skeleton")
    ap.add_argument("--skeleton_dir", default="dataset/ntu_skeleton",
                    help="Folder berisi fall/, not_fall/, file_index.json")
    ap.add_argument("--out_dir", default="docs-pipeline-fix/eda",
                    help="Folder output plot")
    args = ap.parse_args()

    skel = Path(args.skeleton_dir)
    out  = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not skel.exists():
        raise SystemExit(f"[ERROR] Folder tidak ditemukan: {skel}")

    print(f"\n{'#'*70}")
    print(f"# EDA Skeleton Murni (raw extraction)")
    print(f"# Source: {skel}")
    print(f"# Output plot: {out}")
    print(f"{'#'*70}")

    # extraction_stats.json info
    stats_path = skel / "extraction_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            st = json.load(f)
        print(f"\n  Extraction stats: {st}")

    print(f"\nScanning skeleton files...")
    records = scan_skeleton_dir(skel)
    if not records:
        raise SystemExit("[ERROR] Tidak ada file skeleton ditemukan!")

    print_global_stats(records)
    print_per_class_stats(records)
    print_per_action_stats(records)
    print_subject_distribution(records)
    mean_rate = print_joint_rate(records)

    print(f"\n{'='*70}")
    print(f"  GENERATE PLOT")
    print(f"{'='*70}")
    plot_T_distribution(records,        out / "raw_T_distribution.png")
    plot_valid_T_distribution(records,  out / "raw_T_valid_distribution.png")
    plot_T_per_action(records,          out / "raw_T_per_action.png")
    plot_pct_valid(records,             out / "raw_pct_valid.png")
    plot_confidence(records,            out / "raw_confidence.png")
    plot_joint_rate(mean_rate,          out / "raw_joint_rate.png")
    plot_heatmap_relative(records,      out / "raw_heatmap_relative.png")
    plot_T_scatter(records,             out / "raw_T_scatter.png")

    print(f"\n{'='*70}")
    print(f"  EDA SELESAI. Cek folder: {out}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
