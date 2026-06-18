"""
manual_inference_ntu.py
=======================
Jalankan inferensi fall detection pada file .skeleton NTU RGB+D.

Cara pakai:
    python scripts/manual_inference_ntu.py \\
        --skeleton  /path/to/S001C001P001R001A043.skeleton \\
        --weights   work_dir/fall_ntu25_balanced/runs-90-XXXX.pt \\
        --config    config/fall-detection-ntu/balanced.yaml

Atau untuk folder (batch):
    python scripts/manual_inference_ntu.py \\
        --skeleton  /path/to/skeleton_folder/ \\
        --weights   work_dir/fall_ntu25_balanced/runs-90-XXXX.pt \\
        --config    config/fall-detection-ntu/balanced.yaml \\
        --output    results/ntu_predictions.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.prepare_dataset_ntu25 import (
    read_skeleton_file,
    select_primary_body,
    normalize_skeleton,
    pad_or_crop,
    CHANNELS,
    NUM_JOINTS,
)

# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(config_path: str, weights_path: str, device: torch.device):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_args = cfg.get("model_args", {})
    model_class_path = cfg.get("model", "model.BlockGCN.Model")

    components = model_class_path.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    ModelClass = mod

    model = ModelClass(**model_args)
    model = model.to(device)

    weights = torch.load(weights_path, map_location=device)
    if isinstance(weights, dict) and "model_state_dict" in weights:
        weights = weights["model_state_dict"]
    elif isinstance(weights, dict) and "state_dict" in weights:
        weights = weights["state_dict"]

    # Hapus prefix jika ada
    weights = {k.replace("module.", ""): v for k, v in weights.items()}
    model.load_state_dict(weights, strict=False)
    model.eval()
    return model, cfg


# ── Preprocessing satu file ────────────────────────────────────────────────────

def preprocess_skeleton_file(filepath: str, max_frames: int = 150) -> np.ndarray:
    """
    Parse + preprocess satu file .skeleton → tensor (1, 3, T, 25, 1).
    """
    _, frames = read_skeleton_file(filepath)
    seq = select_primary_body(frames)       # (T, 25, 3)
    seq = normalize_skeleton(seq)           # relatif ke SpineBase
    seq = pad_or_crop(seq, max_frames)      # (max_frames, 25, 3)
    tensor = seq.transpose(2, 0, 1)         # (3, max_frames, 25)
    tensor = tensor[:, :, :, np.newaxis]    # (3, max_frames, 25, 1)
    return tensor[np.newaxis].astype(np.float32)  # (1, 3, max_frames, 25, 1)


# ── Inferensi ──────────────────────────────────────────────────────────────────

def run_inference(model, tensor: np.ndarray, device: torch.device):
    """
    tensor: (1, 3, T, 25, 1) numpy float32
    Return: pred_label (int), confidence (float), logits (numpy)
    """
    x = torch.tensor(tensor, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(x)                      # (1, 2)
    probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred   = int(probs.argmax())
    conf   = float(probs[pred])
    return pred, conf, probs


def label_name(label_int: int) -> str:
    return "FALL" if label_int == 1 else "NOT_FALL"


def get_action_class(filename: str) -> int:
    stem = Path(filename).stem
    try:
        a_part = [p for p in stem.split('A') if len(p) >= 3 and p[:3].isdigit()]
        return int(a_part[-1][:3])
    except (IndexError, ValueError):
        return -1


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Inferensi fall detection pada file .skeleton NTU RGB+D")
    ap.add_argument("--skeleton",  required=True,
                    help="File .skeleton atau folder berisi file .skeleton")
    ap.add_argument("--weights",   required=True,
                    help="Path model weights (.pt)")
    ap.add_argument("--config",    required=True,
                    help="Path config YAML")
    ap.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device: 'cuda' atau 'cpu' (default: auto)")
    ap.add_argument("--max_frames", type=int, default=150,
                    help="Jumlah frame per sampel (default 150)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Threshold confidence untuk kelas fall (default 0.5)")
    ap.add_argument("--output",    default=None,
                    help="Simpan hasil ke CSV (opsional)")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"\nDevice   : {device}")
    print(f"Config   : {args.config}")
    print(f"Weights  : {args.weights}")

    # Load model
    print("\nLoading model...")
    model, cfg = load_model(args.config, args.weights, device)
    print("Model siap.")

    # Kumpulkan file skeleton
    skel_path = Path(args.skeleton)
    if skel_path.is_file():
        files = [skel_path]
    elif skel_path.is_dir():
        files = sorted(skel_path.rglob("*.skeleton"))
        print(f"\nDitemukan {len(files)} file .skeleton di {skel_path}")
    else:
        sys.exit(f"[ERROR] Path tidak valid: {args.skeleton}")

    if not files:
        sys.exit("[ERROR] Tidak ada file .skeleton ditemukan.")

    # Jalankan inferensi
    results = []
    correct = 0
    total   = 0
    fall_correct = 0
    fall_total   = 0

    print(f"\n{'─'*65}")
    print(f"{'File':<40} {'GT':>7} {'Pred':>9} {'P(fall)':>9}")
    print(f"{'─'*65}")

    for fp in files:
        try:
            tensor = preprocess_skeleton_file(str(fp), args.max_frames)
            pred, conf, probs = run_inference(model, tensor, device)

            # Gunakan threshold kustom untuk kelas fall
            if probs[1] >= args.threshold:
                final_pred = 1
            else:
                final_pred = 0

            # Ground truth dari nama file
            act = get_action_class(fp.name)
            gt_label = None
            if act == 43:
                gt_label = 1
            elif act in {8, 9, 27, 42}:
                gt_label = 0

            p_fall = float(probs[1])
            status = ""
            if gt_label is not None:
                total += 1
                if gt_label == 1:
                    fall_total += 1
                if final_pred == gt_label:
                    correct += 1
                    if gt_label == 1:
                        fall_correct += 1
                else:
                    status = " ← SALAH"

            gt_str = label_name(gt_label) if gt_label is not None else "?"
            print(f"{fp.name:<40} {gt_str:>7}  {label_name(final_pred):>9}  {p_fall:>8.4f}{status}")
            results.append({
                "file":     fp.name,
                "action":   act,
                "gt":       gt_str,
                "pred":     label_name(final_pred),
                "p_notfall": float(probs[0]),
                "p_fall":   p_fall,
            })
        except Exception as e:
            print(f"{fp.name:<40} [ERROR] {e}")

    # Ringkasan
    print(f"\n{'─'*65}")
    if total > 0:
        acc = correct / total * 100
        print(f"Akurasi keseluruhan : {correct}/{total} = {acc:.1f}%")
        if fall_total > 0:
            sens = fall_correct / fall_total * 100
            print(f"Sensitivity (fall)  : {fall_correct}/{fall_total} = {sens:.1f}%")

    # Simpan CSV jika diminta
    if args.output and results:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nHasil disimpan ke: {out_path}")


if __name__ == "__main__":
    main()
