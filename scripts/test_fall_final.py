"""
test_fall_final.py
===================
Testing model BlockGCN fall detection yang BENAR.

Kunci: menggunakan Feeder yang IDENTIK dengan saat training,
bukan load .npy langsung. Ini memastikan preprocessing konsisten.

Output: file .txt berisi nama_video | label_asli | prediksi | hasil | score_fall

Cara pakai (dari folder scripts/):
    python test_fall_final.py

Atau dengan argumen custom:
    python test_fall_final.py \
        --weights ../work_dir/fall_balanced/runs-26-XXXX.pt \
        --config  ../config/fall-detection/balanced.yaml \
        --data    ../dataset/ntu_data/balanced/joint/val_data.npy \
        --labels  ../dataset/ntu_data/balanced/joint/val_label.pkl \
        --out     ../hasil_test_final.txt
"""

import argparse
import importlib
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Default path — sesuaikan jika perlu ───────────────────────────────────────
DEFAULT_WEIGHTS = "../weights/10/runs-23-3289.pt"
DEFAULT_CONFIG  = "../config/fall-detection/balanced.yaml"
DEFAULT_DATA    = "../dataset/ntu_data/balanced/joint/val_data.npy"
DEFAULT_LABELS  = "../dataset/ntu_data/balanced/joint/val_label.pkl"
DEFAULT_OUT     = "../hasil_test_final_v10.txt"

LABEL_NAMES = {0: "not_fall", 1: "fall"}


# ── Utilities ──────────────────────────────────────────────────────────────────

def import_class(name):
    parts = name.split('.')
    mod   = importlib.import_module('.'.join(parts[:-1]))
    return getattr(mod, parts[-1])


def load_model(cfg, weights_path, device):
    """Load model dari config dan weights file."""
    Model   = import_class(cfg["model"])
    model   = Model(**cfg["model_args"])
    weights = torch.load(weights_path, map_location="cpu")
    weights = {k.replace("module.", ""): v for k, v in weights.items()}
    model.load_state_dict(weights)
    model.to(device).eval()
    return model


def load_dataset(cfg, data_path, labels_path):
    """
    Load dataset menggunakan Feeder yang SAMA dengan training.
    Ini kunci agar preprocessing konsisten — bukan load .npy langsung.
    """
    Feeder      = import_class(cfg["feeder"])
    feeder_args = dict(cfg["test_feeder_args"])

    # Override path jika diberikan
    if data_path:
        feeder_args["data_path"]  = data_path
    if labels_path:
        feeder_args["label_path"] = labels_path

    # Pastikan tidak ada augmentasi saat test
    feeder_args["random_move"]  = False
    feeder_args["random_shift"] = False
    feeder_args["random_flip"]  = False
    feeder_args["random_speed"] = False
    feeder_args["split"]        = "val"
    feeder_args["use_mmap"]     = False

    return Feeder(**feeder_args)


def compute_metrics(labels, preds):
    TP = sum(l == 1 and p == 1 for l, p in zip(labels, preds))
    TN = sum(l == 0 and p == 0 for l, p in zip(labels, preds))
    FP = sum(l == 0 and p == 1 for l, p in zip(labels, preds))
    FN = sum(l == 1 and p == 0 for l, p in zip(labels, preds))
    n    = len(labels)
    acc  = (TP + TN) / n
    prec = TP / max(TP + FP, 1)
    rec  = TP / max(TP + FN, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(TP=TP, TN=TN, FP=FP, FN=FN,
                acc=acc, prec=prec, rec=rec, f1=f1, n=n)


def write_txt(out_path, names, labels, preds, probs_fall,
              metrics, threshold, weights_path, cfg):
    """Tulis hasil test ke file .txt."""
    lines = []

    # Header
    lines.append("# Fall Detection — Hasil Test")
    lines.append("# Tanggal    : {}".format(
        datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    lines.append("# Model      : {}".format(weights_path))
    lines.append("# Threshold  : {:.2f}".format(threshold))
    lines.append("# num_class  : {}  num_point: {}  num_person: {}".format(
        cfg['model_args']['num_class'],
        cfg['model_args']['num_point'],
        cfg['model_args']['num_person']))
    lines.append("# Feeder     : {} (identik dengan training)".format(
        cfg['feeder']))
    lines.append("# " + "─" * 80)

    # Kolom header
    W = 45
    lines.append("# {:<{w}} | {:<10} | {:<10} | {:<7} | score_fall".format(
        "nama_video", "label_asli", "prediksi", "hasil", w=W))
    lines.append("# " + "─" * 80)

    # Per sampel
    for name, label, pred, score in zip(names, labels, preds, probs_fall):
        hasil = "BENAR" if label == pred else "SALAH"
        lines.append("{:<{w}} | {:<10} | {:<10} | {:<7} | {:.4f}".format(
            name,
            LABEL_NAMES[label],
            LABEL_NAMES[pred],
            hasil,
            score,
            w=W
        ))

    # Ringkasan
    m = metrics
    tf = m['TP'] + m['FN']
    tn_total = m['TN'] + m['FP']

    lines.append("")
    lines.append("# " + "─" * 80)
    lines.append("# RINGKASAN")
    lines.append("# " + "─" * 80)
    lines.append("# Total sampel  : {}".format(m['n']))
    lines.append("# Benar         : {}  ({:.2f}%)".format(
        m['TP'] + m['TN'], m['acc'] * 100))
    lines.append("# Salah         : {}".format(m['FP'] + m['FN']))
    lines.append("# " + "─" * 50)
    lines.append("# Accuracy      : {:.2f}%".format(m['acc'] * 100))
    lines.append("# Precision     : {:.2f}%".format(m['prec'] * 100))
    lines.append("# Recall        : {:.2f}%  ← penting untuk safety".format(
        m['rec'] * 100))
    lines.append("# F1-Score      : {:.2f}%".format(m['f1'] * 100))
    lines.append("# " + "─" * 50)
    lines.append("# Confusion Matrix:")
    lines.append("#   TN (not_fall benar)  : {}".format(m['TN']))
    lines.append("#   FP (false alarm)     : {}".format(m['FP']))
    lines.append("#   FN (missed fall)     : {}".format(m['FN']))
    lines.append("#   TP (fall terdeteksi) : {}".format(m['TP']))
    if tf > 0:
        lines.append("# " + "─" * 50)
        lines.append("# Missed fall  : {}/{} = {:.1f}%".format(
            m['FN'], tf, m['FN'] / tf * 100))
        lines.append("# False alarm  : {}/{} = {:.1f}%".format(
            m['FP'], tn_total, m['FP'] / tn_total * 100))

    content = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    return content


def threshold_sweep(labels, probs_fall):
    """Tampilkan metrik pada berbagai threshold."""
    print("\n  Threshold sweep (pilih threshold dengan Recall tinggi):")
    print("  {:>5}  {:>7}  {:>9}  {:>7}  {:>12}".format(
        "Thr", "Acc", "Recall", "F1", "Missed_fall"))
    for thr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        preds = [1 if p >= thr else 0 for p in probs_fall]
        m     = compute_metrics(labels, preds)
        tf    = m['TP'] + m['FN']
        miss  = m['FN'] / max(tf, 1) * 100
        mark  = " ←" if abs(thr - 0.5) < 0.01 else ""
        print("  {:>5.1f}  {:>6.1f}%  {:>8.1f}%  {:>6.1f}%  {:>11.1f}%{}".format(
            thr, m['acc']*100, m['rec']*100, m['f1']*100, miss, mark))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights",    default=DEFAULT_WEIGHTS)
    ap.add_argument("--config",     default=DEFAULT_CONFIG)
    ap.add_argument("--data",       default=DEFAULT_DATA)
    ap.add_argument("--labels",     default=DEFAULT_LABELS)
    ap.add_argument("--out",        default=DEFAULT_OUT)
    ap.add_argument("--threshold",  type=float, default=0.5)
    ap.add_argument("--batch_size", type=int,   default=16)
    ap.add_argument("--device",     default="cuda:0")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA tidak tersedia, pakai CPU")
        args.device = "cpu"

    print("\n" + "=" * 60)
    print("  Fall Detection — Test (pipeline konsisten)")
    print("=" * 60)
    print("  Model    : {}".format(args.weights))
    print("  Threshold: {}".format(args.threshold))
    print("  Output   : {}".format(args.out))

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── 1. Load model ─────────────────────────────────────────────────────────
    print("\nMemuat model ...")
    model = load_model(cfg, args.weights, args.device)
    print("  OK")

    # ── 2. Load dataset via Feeder (SAMA dengan training) ─────────────────────
    print("\nMemuat dataset via Feeder ...")
    dataset = load_dataset(cfg, args.data, args.labels)
    print("  {} sampel".format(len(dataset)))
    print("  Distribusi: {}".format(dataset.class_distribution()))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # ── 3. Inferensi ──────────────────────────────────────────────────────────
    print("\nMenjalankan inferensi ...")
    all_labels = []
    all_preds  = []
    all_probs  = []

    for batch_idx, (data, label) in enumerate(loader):
        data = data.float().to(args.device)

        with torch.no_grad():
            output = model(data)                          # (B, 2)
            probs  = torch.softmax(output, dim=1)        # (B, 2)
            pred   = torch.argmax(probs, dim=1)          # (B,)

        all_labels.extend(label.numpy().tolist())
        all_preds.extend(pred.cpu().numpy().tolist())
        all_probs.extend(probs[:, 1].cpu().numpy().tolist())

        done = min((batch_idx + 1) * args.batch_size, len(dataset))
        print("\r  {}/{}".format(done, len(dataset)), end="", flush=True)

    print()

    # ── 4. Hitung metrik ──────────────────────────────────────────────────────
    probs_fall = all_probs
    preds      = [1 if p >= args.threshold else 0 for p in probs_fall]
    metrics    = compute_metrics(all_labels, preds)

    # ── 5. Tulis .txt ─────────────────────────────────────────────────────────
    names = dataset.sample_name
    print("\nMenulis ke: {}".format(args.out))
    write_txt(
        out_path    = args.out,
        names       = names,
        labels      = all_labels,
        preds       = preds,
        probs_fall  = probs_fall,
        metrics     = metrics,
        threshold   = args.threshold,
        weights_path= args.weights,
        cfg         = cfg,
    )

    # ── 6. Ringkasan terminal ─────────────────────────────────────────────────
    m = metrics
    print("\n" + "=" * 60)
    print("  HASIL TEST")
    print("=" * 60)
    print("  Total sampel : {}".format(m['n']))
    print("  Accuracy     : {:.2f}%".format(m['acc']  * 100))
    print("  Precision    : {:.2f}%".format(m['prec'] * 100))
    print("  Recall       : {:.2f}%".format(m['rec']  * 100))
    print("  F1-Score     : {:.2f}%".format(m['f1']   * 100))
    print("\n  Confusion Matrix:")
    print("               Pred NOT_FALL  Pred FALL")
    print("  True NOT_FALL     {:^5}         {:^5}".format(m['TN'], m['FP']))
    print("  True FALL         {:^5}         {:^5}".format(m['FN'], m['TP']))
    tf = m['TP'] + m['FN']
    tn = m['TN'] + m['FP']
    if tf > 0:
        print("\n  Missed fall  : {}/{} = {:.1f}%".format(
            m['FN'], tf, m['FN']/tf*100))
        print("  False alarm  : {}/{} = {:.1f}%".format(
            m['FP'], tn, m['FP']/tn*100))

    print("\n  Tersimpan di : {}".format(args.out))

    # ── 7. Threshold sweep ────────────────────────────────────────────────────
    threshold_sweep(all_labels, probs_fall)
    print()


if __name__ == "__main__":
    main()