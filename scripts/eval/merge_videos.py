#!/usr/bin/env python3
"""
merge_videos.py - Menggabungkan beberapa video menjadi satu video

Cara pakai:
    python merge_videos.py --folder /path/ke/folder --count 5
    python merge_videos.py --folder /path/ke/folder --count 5 --format avi
    python merge_videos.py --folder /path/ke/folder --count 5 --output hasil.mp4
    python merge_videos.py --folder /path/ke/folder --count 5 --output hasil.mp4 --sort name
    python merge_videos.py --folder /path/ke/folder --files video1.mp4 video3.mp4 video2.mp4

Format output yang didukung:
    mp4  → codec: libx264 + aac  (paling kompatibel, default)
    avi  → codec: mpeg4 + mp3    (legacy, ukuran lebih besar)
    mkv  → codec: libx264 + aac  (container fleksibel)
    mov  → codec: libx264 + aac  (Apple QuickTime)
    webm → codec: libvpx + libvorbis (web-friendly)

Dependensi:
    pip install moviepy
    (opsional, lebih cepat) apt install ffmpeg  atau  brew install ffmpeg
"""

import argparse
import os
import sys
from pathlib import Path


# ─── Ekstensi video yang didukung ─────────────────────────────────────────────
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}

# ─── Konfigurasi codec per format output ──────────────────────────────────────
# (video_codec_moviepy, audio_codec_moviepy, video_codec_ffmpeg, audio_codec_ffmpeg)
FORMAT_CONFIG = {
    "mp4":  ("libx264",     "aac",         "libx264",  "aac"),
    "avi":  ("mpeg4",       "libmp3lame",  "mpeg4",    "libmp3lame"),
    "mkv":  ("libx264",     "aac",         "libx264",  "aac"),
    "mov":  ("libx264",     "aac",         "libx264",  "aac"),
    "webm": ("libvpx",      "libvorbis",   "libvpx",   "libvorbis"),
}


def get_video_files(folder: Path, count: int, sort_by: str) -> list[Path]:
    """Ambil file video dari folder, diurutkan sesuai pilihan, lalu batasi sejumlah count."""
    files = [f for f in folder.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS]

    if not files:
        print(f"[ERROR] Tidak ada file video di folder: {folder}")
        sys.exit(1)

    # Sorting
    if sort_by == "name":
        files.sort(key=lambda f: f.name.lower())
    elif sort_by == "date":
        files.sort(key=lambda f: f.stat().st_mtime)
    elif sort_by == "size":
        files.sort(key=lambda f: f.stat().st_size)
    # "none" → urutan bawaan sistem

    if count > len(files):
        print(f"[WARNING] Hanya ada {len(files)} video, akan digabung semua.")
        count = len(files)

    selected = files[:count]
    print(f"\n[INFO] Video yang akan digabung ({len(selected)} file):")
    for i, f in enumerate(selected, 1):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {i}. {f.name}  ({size_mb:.1f} MB)")
    return selected


def merge_with_moviepy(video_paths: list[Path], output_path: Path, fmt: str) -> None:
    """Gabungkan video menggunakan moviepy."""
    try:
        from moviepy.editor import VideoFileClip, concatenate_videoclips
    except ImportError:
        print("[ERROR] moviepy belum terinstall. Jalankan: pip install moviepy")
        sys.exit(1)

    video_codec, audio_codec, _, _ = FORMAT_CONFIG[fmt]

    print("\n[INFO] Memuat video...")
    clips = []
    try:
        for path in video_paths:
            print(f"  → Loading: {path.name}")
            clip = VideoFileClip(str(path))
            clips.append(clip)

        print("\n[INFO] Menggabungkan video...")
        final = concatenate_videoclips(clips, method="compose")

        print(f"[INFO] Format output : .{fmt}  (codec: {video_codec} + {audio_codec})")
        print(f"[INFO] Menyimpan ke  : {output_path}")
        final.write_videofile(
            str(output_path),
            codec=video_codec,
            audio_codec=audio_codec,
            logger="bar",
        )
    finally:
        for clip in clips:
            clip.close()

    print(f"\n[✓] Selesai! File tersimpan: {output_path}")
    print(f"    Ukuran output: {output_path.stat().st_size / (1024*1024):.1f} MB")


def merge_with_ffmpeg(video_paths: list[Path], output_path: Path, fmt: str) -> None:
    """Gabungkan video menggunakan ffmpeg langsung (lebih cepat)."""
    import subprocess
    import tempfile

    _, _, video_codec, audio_codec = FORMAT_CONFIG[fmt]

    # Buat file list sementara
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        list_file = f.name
        for path in video_paths:
            safe_path = str(path.resolve()).replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")

    try:
        print(f"\n[INFO] Format output : .{fmt}  (codec: {video_codec} + {audio_codec})")
        print(f"[INFO] Menggabungkan & menyimpan ke: {output_path}")

        # Coba dulu -c copy (cepat, tanpa re-encode) — hanya works kalau format sama
        cmd_copy = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            str(output_path),
        ]
        result = subprocess.run(cmd_copy, capture_output=True)

        if result.returncode != 0:
            print("[INFO] Stream copy tidak bisa, melakukan re-encode...")
            cmd_encode = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", list_file,
                "-vcodec", video_codec,
                "-acodec", audio_codec,
                str(output_path),
            ]
            subprocess.run(cmd_encode, check=True)
    finally:
        os.unlink(list_file)

    print(f"\n[✓] Selesai! File tersimpan: {output_path}")
    print(f"    Ukuran output: {output_path.stat().st_size / (1024*1024):.1f} MB")


def check_ffmpeg() -> bool:
    """Cek apakah ffmpeg tersedia di PATH."""
    import shutil
    return shutil.which("ffmpeg") is not None


# ─── Argparse ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Gabungkan beberapa video menjadi satu file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Format output yang tersedia:
  mp4   → libx264 + aac        (default, paling kompatibel)
  avi   → mpeg4  + libmp3lame  (legacy player)
  mkv   → libx264 + aac        (container fleksibel)
  mov   → libx264 + aac        (Apple QuickTime)
  webm  → libvpx  + libvorbis  (web browser)

Contoh:
  # Output MP4 (default)
  python merge_videos.py --folder ./videos --count 5

  # Output AVI
  python merge_videos.py --folder ./videos --count 5 --format avi

  # Output MKV, urut berdasarkan tanggal
  python merge_videos.py --folder ./videos --count 3 --format mkv --sort date

  # Pilih file manual, simpan sebagai MOV
  python merge_videos.py --folder ./videos --files clip1.mp4 clip2.mp4 --format mov

  # Tentukan nama output sendiri (ekstensi otomatis mengikuti --format)
  python merge_videos.py --folder ./videos --count 5 --format avi --output gabungan
        """,
    )

    parser.add_argument(
        "--folder", "-f",
        type=Path,
        required=True,
        help="Path ke folder yang berisi video.",
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=None,
        help="Jumlah video yang akan digabung (diambil dari urutan awal setelah sorting).",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Daftar nama file spesifik (di dalam --folder) yang ingin digabung, sesuai urutan.",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Nama file output (default: merged_output.mp4 di dalam --folder).",
    )
    parser.add_argument(
        "--sort", "-s",
        choices=["name", "date", "size", "none"],
        default="name",
        help="Urutan pengambilan video: name (default), date, size, none.",
    )
    parser.add_argument(
        "--format", "-fmt",
        choices=list(FORMAT_CONFIG.keys()),
        default="mp4",
        help="Format file output: mp4 (default), avi, mkv, mov, webm.",
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "ffmpeg", "moviepy"],
        default="auto",
        help="Engine yang dipakai: auto (ffmpeg jika tersedia, else moviepy), ffmpeg, moviepy.",
    )

    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Validasi folder
    folder: Path = args.folder.resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"[ERROR] Folder tidak ditemukan: {folder}")
        sys.exit(1)

    # Tentukan video yang akan digabung
    if args.files:
        # Mode manual: user tentukan file sendiri
        video_paths = []
        for name in args.files:
            p = folder / name
            if not p.exists():
                print(f"[ERROR] File tidak ditemukan: {p}")
                sys.exit(1)
            video_paths.append(p)
        print(f"\n[INFO] Video yang akan digabung ({len(video_paths)} file):")
        for i, f in enumerate(video_paths, 1):
            print(f"  {i}. {f.name}")
    else:
        if args.count is None:
            print("[ERROR] Tentukan --count atau gunakan --files.")
            sys.exit(1)
        video_paths = get_video_files(folder, args.count, args.sort)

    fmt: str = args.format  # "mp4", "avi", "mkv", "mov", "webm"

    # Tentukan output path — paksa ekstensi sesuai --format
    if args.output:
        base = args.output if args.output.is_absolute() else folder / args.output
        # Ganti ekstensi kalau user lupa / beda dengan --format
        output_path = base.with_suffix(f".{fmt}")
        if base.suffix and base.suffix.lower() != f".{fmt}":
            print(f"[INFO] Ekstensi output disesuaikan menjadi .{fmt} → {output_path.name}")
    else:
        output_path = folder / f"merged_output.{fmt}"

    if output_path.exists():
        confirm = input(f"\n[WARNING] File {output_path} sudah ada. Timpa? (y/n): ").strip().lower()
        if confirm != "y":
            print("Dibatalkan.")
            sys.exit(0)

    # Pilih engine
    use_ffmpeg = False
    if args.engine == "ffmpeg":
        if not check_ffmpeg():
            print("[ERROR] ffmpeg tidak ditemukan. Install dulu atau pakai --engine moviepy.")
            sys.exit(1)
        use_ffmpeg = True
    elif args.engine == "moviepy":
        use_ffmpeg = False
    else:  # auto
        use_ffmpeg = check_ffmpeg()
        print(f"[INFO] Engine: {'ffmpeg (ditemukan di PATH)' if use_ffmpeg else 'moviepy (ffmpeg tidak ada)'}")

    # Jalankan merge
    if use_ffmpeg:
        merge_with_ffmpeg(video_paths, output_path, fmt)
    else:
        merge_with_moviepy(video_paths, output_path, fmt)


if __name__ == "__main__":
    main()
