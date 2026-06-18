"""
merge_val_videos.py
===================
Gabungkan SELURUH video set-uji (val) NTU menjadi satu video kontinu, untuk
analisis perilaku realtime (latency deteksi, false-alarm/menit, FPS stabil) dan
demo dengan scripts/realtime_video_inference.py.

Daftar + URUTAN video diambil dari val_label.pkl (set uji yang sama dgn tabel
akurasi), dipetakan ke <video_dir>/<nama>.avi.

Output (default di dataset/ntu_videos/):
  merged_val.mp4            -> video gabungan (concat -c copy, tanpa re-encode)
  merged_val_concat.txt     -> daftar concat ffmpeg (bisa diaudit)
  merged_val_timeline.csv   -> GROUND TRUTH per-frame:
                               idx,name,label,n_frames,start_frame,end_frame
                               (start/end = indeks frame di video gabungan)

Timeline inilah yang nanti dipakai untuk evaluasi event-based pada stream
kontinu (recall kejadian, jeda deteksi, false alarm/menit).

Catatan: semua video val seragam (mpeg4, 1920x1080, 30fps) sehingga concat
-c copy aman & cepat. Pakai --reencode hanya bila ffmpeg mengeluh.

Urutan default = 'action' (ROUND-ROBIN / bergiliran): 1 putaran berisi
A008, A009, A027, A042, A043, lalu balik lagi A008, A009, A027, A042, A043,
dst. Jadi fall (A043) muncul berkala di sepanjang stream, bukan menumpuk di
akhir. Bila satu aksi habis, aksi itu dilewati di putaran berikutnya.

Cara pakai (kamu yang run):
    python scripts/merge_val_videos.py                        # bergiliran 8,9,27,42,43 berulang
    python scripts/merge_val_videos.py --order pkl            # urutan asli (berselang-seling)
    python scripts/merge_val_videos.py --per-class 30         # subset 30/kelas utk uji cepat
    python scripts/merge_val_videos.py --reencode             # bila -c copy gagal
"""

import argparse
import csv
import pickle
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

DEFAULT_LABEL_PKL = "dataset/yolo17_data/full/joint/val_label.pkl"
DEFAULT_VIDEO_DIR = "dataset/ntu_videos"
DEFAULT_OUT       = "dataset/ntu_videos/merged_val.mp4"
DEFAULT_EXT       = ".avi"

LABEL_NAMES = {0: "not_fall", 1: "fall"}

# Urutan aksi NTU yang dipakai (8,9,27,42 = not_fall ; 43 = fall)
ACTION_ORDER = [8, 9, 27, 42, 43]
ACTION_RANK = {a: i for i, a in enumerate(ACTION_ORDER)}


def get_action_code(name):
    """S001C001P003R001A008_rgb -> 8 (None jika tidak ketemu)."""
    m = re.search(r'A(\d+)', name)
    return int(m.group(1)) if m else None


def probe_nframes(video_path):
    """Jumlah frame video via ffprobe. Coba header dulu (cepat),
    fallback hitung-frame (lambat tapi andal)."""
    # Cara cepat: nb_frames dari header (tersedia utk mpeg4 NTU)
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames",
             "-of", "default=nw=1:nk=1", str(video_path)],
            capture_output=True, text=True, timeout=60)
        val = out.stdout.strip()
        if val.isdigit() and int(val) > 0:
            return int(val)
    except Exception:
        pass
    # Fallback: dekode & hitung frame sebenarnya (lambat)
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_frames", "-show_entries", "stream=nb_read_frames",
             "-of", "default=nw=1:nk=1", str(video_path)],
            capture_output=True, text=True, timeout=120)
        val = out.stdout.strip()
        if val.isdigit():
            return int(val)
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser(description="Gabungkan semua video val jadi satu stream")
    ap.add_argument("--label-pkl", default=DEFAULT_LABEL_PKL)
    ap.add_argument("--video-dir", default=DEFAULT_VIDEO_DIR)
    ap.add_argument("--out",       default=DEFAULT_OUT)
    ap.add_argument("--ext",       default=DEFAULT_EXT)
    ap.add_argument("--order",     default="action",
                    choices=["action", "pkl", "grouped", "shuffle"],
                    help="action=urut kode aksi 8,9,27,42,43 (default) | "
                         "pkl=urutan asli (kelas berselang-seling) | "
                         "grouped=not_fall dulu lalu fall | shuffle=acak")
    ap.add_argument("--per-class", type=int, default=0,
                    help="Ambil N video/kelas saja (0=semua)")
    ap.add_argument("--per-action", type=int, default=0,
                    help="Ambil N video/aksi (8,9,27,42,43) saja, dipakai "
                         "dgn --order action. N=jumlah putaran round-robin. "
                         "(0=semua)")
    ap.add_argument("--limit",     type=int, default=0,
                    help="Batasi total N video pertama (0=semua)")
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--reencode",  action="store_true",
                    help="Re-encode (lambat) bila -c copy gagal")
    ap.add_argument("--no-timeline", action="store_true",
                    help="Lewati pembuatan timeline (skip ffprobe per-klip)")
    args = ap.parse_args()

    label_pkl = ROOT / args.label_pkl
    video_dir = ROOT / args.video_dir
    out_path  = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Daftar video val ─────────────────────────────────────────────────────
    with open(label_pkl, "rb") as f:
        names, labels = pickle.load(f)
    samples = []
    missing = 0
    for name, label in zip(names, labels):
        vp = video_dir / (name + args.ext)
        if vp.exists():
            samples.append((name, int(label), vp))
        else:
            missing += 1
    if missing:
        print(f"[WARN] {missing} video tidak ditemukan di {video_dir}, dilewati.")

    # ── per-class subset ─────────────────────────────────────────────────────
    if args.per_class > 0:
        rng = random.Random(args.seed)
        by_cls = {0: [], 1: []}
        for s in samples:
            by_cls[s[1]].append(s)
        sub = []
        for c in (0, 1):
            pool = by_cls[c]
            rng.shuffle(pool)
            sub.extend(pool[:args.per_class])
        samples = sub

    # ── Urutan ───────────────────────────────────────────────────────────────
    if args.order == "action":
        # ROUND-ROBIN: 1 putaran = 8,9,27,42,43 lalu ulang ke 8,9,27,42,43 ...
        buckets = {a: [] for a in ACTION_ORDER}
        for s in samples:
            a = get_action_code(s[0])
            if a in buckets:
                buckets[a].append(s)
        for a in buckets:
            buckets[a].sort(key=lambda s: s[0])
        if args.per_action > 0:
            for a in buckets:
                buckets[a] = buckets[a][:args.per_action]
        interleaved, idx = [], {a: 0 for a in ACTION_ORDER}
        while True:
            added = False
            for a in ACTION_ORDER:        # urutan dalam 1 putaran
                if idx[a] < len(buckets[a]):
                    interleaved.append(buckets[a][idx[a]])
                    idx[a] += 1
                    added = True
            if not added:
                break
        samples = interleaved
    elif args.order == "grouped":
        samples.sort(key=lambda s: (s[1], s[0]))
    elif args.order == "shuffle":
        random.Random(args.seed).shuffle(samples)
    # pkl: biarkan urutan asli

    if args.limit > 0:
        samples = samples[:args.limit]

    if not samples:
        sys.exit("[ERROR] Tidak ada video untuk digabung.")

    n_fall = sum(s[1] for s in samples)
    print(f"\n{'='*60}")
    print("  MERGE VIDEO VAL")
    print(f"{'='*60}")
    print(f"  Total video : {len(samples)}  (not_fall={len(samples)-n_fall}, fall={n_fall})")
    print(f"  Urutan      : {args.order}")
    if args.order == "action":
        per_act = {}
        for name, _, _ in samples:
            per_act[get_action_code(name)] = per_act.get(get_action_code(name), 0) + 1
        seg = "  ".join(f"A{a:03d}={per_act.get(a,0)}" for a in ACTION_ORDER)
        print(f"  Per aksi    : {seg}")
    print(f"  Output      : {out_path}")

    # ── Concat list (forward-slash, single-quoted utk ffmpeg concat demuxer) ──
    concat_txt = out_path.with_name(out_path.stem + "_concat.txt")
    with open(concat_txt, "w", encoding="utf-8") as f:
        for _, _, vp in samples:
            p = str(vp.resolve()).replace("\\", "/").replace("'", "'\\''")
            f.write(f"file '{p}'\n")
    print(f"  Concat list : {concat_txt}")

    # ── Timeline ground-truth (frame boundaries) ─────────────────────────────
    if not args.no_timeline:
        print(f"\n  Membuat timeline (ffprobe per-klip, {len(samples)} video)...")
        timeline_csv = out_path.with_name(out_path.stem + "_timeline.csv")
        cursor = 0
        rows = []
        n_bad = 0
        for i, (name, label, vp) in enumerate(samples, 1):
            nf = probe_nframes(vp)
            if nf is None:
                nf = 0
                n_bad += 1
            start = cursor
            end = cursor + nf - 1 if nf > 0 else cursor
            rows.append((i - 1, name, label, LABEL_NAMES[label], nf, start, end))
            cursor += nf
            if i % 100 == 0 or i == len(samples):
                print(f"\r    [{i}/{len(samples)}] frame kumulatif: {cursor}", end="", flush=True)
        print()
        with open(timeline_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx", "name", "label", "label_name", "n_frames",
                        "start_frame", "end_frame"])
            w.writerows(rows)
        total_frames = cursor
        dur_s = total_frames / 30.0
        print(f"  Timeline    : {timeline_csv}")
        print(f"  Total frame : {total_frames}  (~{dur_s/60:.1f} menit @30fps)")
        if n_bad:
            print(f"  [WARN] {n_bad} klip gagal diprobe (n_frames=0 di timeline)")

    # ── Jalankan ffmpeg concat ───────────────────────────────────────────────
    print(f"\n  Menjalankan ffmpeg concat...")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
           "-i", str(concat_txt)]
    if args.reencode:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-an", str(out_path)]
    else:
        cmd += ["-c", "copy", "-an", str(out_path)]

    print("  " + " ".join(cmd))
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print("\n[ERROR] ffmpeg gagal.")
        if not args.reencode:
            print("  Coba ulang dengan --reencode (lebih lambat, lebih kompatibel).")
        sys.exit(1)

    size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0
    print(f"\n{'='*60}")
    print(f"  SELESAI -> {out_path}  ({size_mb:.0f} MB)")
    print(f"{'='*60}")
    print("\n  Demo realtime + FPS di layar:")
    print(f"    python scripts/realtime_video_inference.py --video {args.out} --device cuda:0")


if __name__ == "__main__":
    main()