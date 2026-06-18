"""
eda_video_frames.py
===================
Scan semua video di folder NTU dan tampilkan distribusi jumlah frame.

Output:
  - Console: stats per class (fall vs not_fall + per action A008/A009/A027/A042/A043)
  - CSV    : daftar file + frame_count + fps + duration_sec
  - Plot   : histogram + boxplot distribusi frame count

Cara pakai:
    python scripts/eda_video_frames.py
    python scripts/eda_video_frames.py --video_dir dataset/ntu_videos
    python scripts/eda_video_frames.py --out_dir docs-pipeline-fix/eda --no_plot
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

VIDEO_EXT = {".avi", ".mp4", ".mov", ".mkv"}

FALL_CLASSES     = {43}
NOT_FALL_CLASSES = {8, 9, 27, 42}
ALL_CLASSES      = FALL_CLASSES | NOT_FALL_CLASSES

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

CLASS_NAMES  = {0: "not_fall", 1: "fall"}
CLASS_COLORS = {0: "#2ecc71", 1: "#e74c3c"}


def parse_action(stem):
    m = re.search(r'[Aa](\d{3})', stem)
    return int(m.group(1)) if m else None


def parse_subject(stem):
    m = re.search(r'[Pp](\d{3})', stem)
    return int(m.group(1)) if m else None


def label_of(action):
    if action in FALL_CLASSES:
        return 1
    if action in NOT_FALL_CLASSES:
        return 0
    return -1


def probe_video(path: Path):
    """Return (frame_count, fps, duration_sec) atau None kalau gagal."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if n <= 0:
        return None
    dur = n / fps if fps > 0 else 0.0
    return {"frames": n, "fps": fps, "duration": dur, "w": w, "h": h}


# ── Stats printer ──────────────────────────────────────────────────────────────

def print_stats_block(name, frames):
    arr = np.asarray(frames)
    print(f"  {name:18s}  n={len(arr):4d}  "
          f"min={int(arr.min()):3d}  max={int(arr.max()):3d}  "
          f"mean={arr.mean():6.1f}  median={np.median(arr):5.1f}  "
          f"std={arr.std():5.1f}  "
          f"p25={np.percentile(arr,25):.0f}  p75={np.percentile(arr,75):.0f}")


def print_full_report(records):
    print(f"\n{'='*78}")
    print(f"  STATISTIK GLOBAL")
    print(f"{'='*78}")

    Ns = [r["frames"] for r in records]
    Ds = [r["duration"] for r in records]
    Fs = [r["fps"] for r in records]
    print(f"  Total video        : {len(records)}")
    print(f"  Frame count        : min={min(Ns)}  max={max(Ns)}  "
          f"mean={np.mean(Ns):.1f}  median={np.median(Ns):.0f}")
    print(f"  FPS (median)       : {np.median(Fs):.1f}")
    print(f"  Durasi (detik)     : min={min(Ds):.1f}  max={max(Ds):.1f}  "
          f"mean={np.mean(Ds):.1f}  median={np.median(Ds):.1f}")

    print(f"\n{'─'*78}")
    print(f"  PER LABEL (fall vs not_fall)")
    print(f"{'─'*78}")
    for lbl in (0, 1):
        sub = [r["frames"] for r in records if r["label"] == lbl]
        if not sub:
            continue
        print_stats_block(CLASS_NAMES[lbl], sub)

    print(f"\n{'─'*78}")
    print(f"  PER ACTION CLASS NTU")
    print(f"{'─'*78}")
    groups = defaultdict(list)
    for r in records:
        if r["action"] is not None:
            groups[r["action"]].append(r["frames"])
    for cls in sorted(groups.keys()):
        name = ACTION_NAMES.get(cls, "unknown")
        print_stats_block(f"A{cls:03d} {name}", groups[cls])

    other = [r for r in records if r["action"] not in ALL_CLASSES and r["action"] is not None]
    if other:
        print(f"\n  [INFO] {len(other)} video dengan action class di luar A008/A009/A027/A042/A043")

    no_action = [r for r in records if r["action"] is None]
    if no_action:
        print(f"  [INFO] {len(no_action)} video tanpa pattern Axxx di nama")


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_frames_per_class(records, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    all_frames = [r["frames"] for r in records]
    max_n = max(all_frames)
    bins = np.linspace(0, max_n + 5, 31)

    ax = axes[0]
    for lbl in (0, 1):
        sub = [r["frames"] for r in records if r["label"] == lbl]
        if sub:
            ax.hist(sub, bins=bins, alpha=0.6, color=CLASS_COLORS[lbl],
                    label=f"{CLASS_NAMES[lbl]} (n={len(sub)}, mean={np.mean(sub):.0f})",
                    edgecolor="black")
    ax.set_xlabel("Jumlah frame per video")
    ax.set_ylabel("Frekuensi")
    ax.set_title("Distribusi frame count — fall vs not_fall")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    data_box = [[r["frames"] for r in records if r["label"] == lbl] for lbl in (0, 1)]
    lbl_box = [CLASS_NAMES[lbl] for lbl in (0, 1)]
    bp = ax.boxplot(data_box, labels=lbl_box, patch_artist=True, showmeans=True)
    for patch, lbl in zip(bp["boxes"], (0, 1)):
        patch.set_facecolor(CLASS_COLORS[lbl])
        patch.set_alpha(0.6)
    ax.set_ylabel("Frame count")
    ax.set_title("Boxplot per class")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot tersimpan: {out_path}")


def plot_frames_per_action(records, out_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    groups = defaultdict(list)
    for r in records:
        if r["action"] is not None:
            groups[r["action"]].append(r["frames"])
    keys = sorted(groups.keys())
    data = [groups[k] for k in keys]
    labels_x = [f"A{k:03d}\n{ACTION_NAMES.get(k,'?')}\nn={len(groups[k])}" for k in keys]
    colors = [ACTION_COLORS.get(k, "#7f8c8d") for k in keys]

    bp = ax.boxplot(data, labels=labels_x, patch_artist=True, showmeans=True)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
    ax.set_ylabel("Frame count")
    ax.set_title("Boxplot frame count per NTU action class")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot tersimpan: {out_path}")


def plot_duration(records, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    all_dur = [r["duration"] for r in records]
    max_d = max(all_dur)
    bins = np.linspace(0, max_d + 0.5, 31)
    for lbl in (0, 1):
        sub = [r["duration"] for r in records if r["label"] == lbl]
        if sub:
            ax.hist(sub, bins=bins, alpha=0.6, color=CLASS_COLORS[lbl],
                    label=f"{CLASS_NAMES[lbl]} (mean={np.mean(sub):.2f}s)",
                    edgecolor="black")
    ax.set_xlabel("Durasi video (detik)")
    ax.set_ylabel("Frekuensi")
    ax.set_title("Distribusi durasi video")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot tersimpan: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="EDA frame count video NTU")
    ap.add_argument("--video_dir", default="dataset/ntu_videos")
    ap.add_argument("--out_dir",   default="docs-pipeline-fix/eda")
    ap.add_argument("--csv_name",  default="video_frames.csv")
    ap.add_argument("--no_plot",   action="store_true")
    args = ap.parse_args()

    vdir = Path(args.video_dir)
    out  = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not vdir.exists():
        raise SystemExit(f"[ERROR] Folder tidak ditemukan: {vdir}")

    files = sorted(p for p in vdir.iterdir() if p.suffix.lower() in VIDEO_EXT)
    if not files:
        raise SystemExit(f"[ERROR] Tidak ada video di {vdir}")

    print(f"\n{'#'*78}")
    print(f"# EDA Frame Count Video NTU")
    print(f"# Source: {vdir}  ({len(files)} file)")
    print(f"# Output: {out}")
    print(f"{'#'*78}\n")

    records = []
    failed  = 0
    for i, fp in enumerate(files, 1):
        info = probe_video(fp)
        if info is None:
            failed += 1
            continue
        action = parse_action(fp.stem)
        records.append({
            "name":     fp.name,
            "stem":     fp.stem,
            "action":   action,
            "subject":  parse_subject(fp.stem),
            "label":    label_of(action),
            "frames":   info["frames"],
            "fps":      info["fps"],
            "duration": info["duration"],
            "w":        info["w"],
            "h":        info["h"],
        })
        if i % 500 == 0:
            print(f"  [{i}/{len(files)}] diproses...")

    if failed:
        print(f"\n  [WARN] {failed} video gagal dibaca, dilewati.")

    if not records:
        raise SystemExit("[ERROR] Tidak ada video berhasil dibaca!")

    # Save CSV
    csv_path = out / args.csv_name
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "name", "stem", "action", "subject", "label",
            "frames", "fps", "duration", "w", "h"])
        w.writeheader()
        w.writerows(records)
    print(f"\n  CSV tersimpan: {csv_path}")

    # Report
    print_full_report(records)

    # Plots
    if args.no_plot:
        return
    if not HAS_MPL:
        print("\n[WARN] matplotlib tidak ada, skip plot. pip install matplotlib")
        return

    print(f"\n{'='*78}")
    print(f"  GENERATE PLOT")
    print(f"{'='*78}")
    plot_frames_per_class(records,  out / "video_frames_per_class.png")
    plot_frames_per_action(records, out / "video_frames_per_action.png")
    plot_duration(records,          out / "video_duration.png")

    print(f"\n{'='*78}")
    print(f"  EDA SELESAI. Output di: {out}")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
