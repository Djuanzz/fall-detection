"""
test_fall_ntu25.py
==================
Testing model BlockGCN fall detection pada dataset NTU RGB+D 25-joint.

Menggunakan Feeder yang IDENTIK dengan saat training (feeder_ntu_binary)
sehingga preprocessing konsisten — bukan load .npy langsung.

Metrik yang dihitung:
  - Accuracy, Balanced Accuracy
  - Precision, Recall (Sensitivity), Specificity
  - F1-Score, AUC-ROC
  - Confusion Matrix + Threshold Sweep

Output: file .txt berisi nama_sampel | label_asli | prediksi | hasil | score_fall

Cara pakai (dari root project):
    python scripts/test_fall_ntu25.py

Dengan argumen custom:
    python scripts/test_fall_ntu25.py \\
        --weights work_dir/fall_ntu25_balanced/runs-90-XXXX.pt \\
        --config  config/fall-detection-ntu/balanced.yaml \\
        --data    dataset/ntu25_data/balanced/joint/val_data.npy \\
        --labels  dataset/ntu25_data/balanced/joint/val_label.pkl \\
        --out     hasil_test_ntu25.txt
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

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# ── Default paths — sesuaikan jika perlu ──────────────────────────────────────
DEFAULT_WEIGHTS = "work_dir/fall_ntu25_balanced/runs-82-15498.pt"
DEFAULT_CONFIG  = "config/fall-detection-ntu/balanced.yaml"
DEFAULT_DATA    = "dataset/ntu25_data/balanced/joint/val_data.npy"
DEFAULT_LABELS  = "dataset/ntu25_data/balanced/joint/val_label.pkl"
DEFAULT_OUT     = "hasil_test_ntu25.txt"

LABEL_NAMES = {0: "not_fall", 1: "fall"}


# ── Utilities ──────────────────────────────────────────────────────────────────

def import_class(name):
    parts = name.split('.')
    mod   = importlib.import_module('.'.join(parts[:-1]))
    return getattr(mod, parts[-1])


def load_model(cfg, weights_path, device):
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
    Semua augmentasi dimatikan untuk test.
    """
    Feeder      = import_class(cfg["feeder"])
    feeder_args = dict(cfg["test_feeder_args"])

    if data_path:
        feeder_args["data_path"]  = data_path
    if labels_path:
        feeder_args["label_path"] = labels_path

    feeder_args["random_move"]  = False
    feeder_args["random_shift"] = False
    feeder_args["random_flip"]  = False
    feeder_args["random_speed"] = False
    feeder_args["random_noise"] = False
    feeder_args["split"]        = "val"
    feeder_args["use_mmap"]     = False

    return Feeder(**feeder_args)


# ── Metrik ─────────────────────────────────────────────────────────────────────

def compute_metrics(labels, preds, probs_fall):
    TP = sum(l == 1 and p == 1 for l, p in zip(labels, preds))
    TN = sum(l == 0 and p == 0 for l, p in zip(labels, preds))
    FP = sum(l == 0 and p == 1 for l, p in zip(labels, preds))
    FN = sum(l == 1 and p == 0 for l, p in zip(labels, preds))
    n = len(labels)

    acc         = (TP + TN) / n
    prec        = TP / max(TP + FP, 1)
    sensitivity = TP / max(TP + FN, 1)   # Recall
    specificity = TN / max(TN + FP, 1)
    f1          = 2 * prec * sensitivity / max(prec + sensitivity, 1e-9)
    bal_acc     = (sensitivity + specificity) / 2.0

    # AUC-ROC dengan trapezoidal rule (tanpa sklearn)
    auc = _compute_auc(labels, probs_fall)

    return dict(
        TP=TP, TN=TN, FP=FP, FN=FN, n=n,
        acc=acc, bal_acc=bal_acc,
        prec=prec, sensitivity=sensitivity, specificity=specificity,
        f1=f1, auc=auc,
    )


def _compute_auc(labels, scores):
    """AUC-ROC via trapezoidal rule (tidak butuh sklearn)."""
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp, fp = 0, 0
    roc_points = [(0.0, 0.0)]
    prev_score = None
    for score, label in pairs:
        if prev_score is not None and score != prev_score:
            roc_points.append((fp / n_neg, tp / n_pos))
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    roc_points.append((1.0, 1.0))

    auc = 0.0
    for i in range(1, len(roc_points)):
        x1, y1 = roc_points[i - 1]
        x2, y2 = roc_points[i]
        auc += (x2 - x1) * (y1 + y2) / 2.0
    return auc


# ── Output .txt ────────────────────────────────────────────────────────────────

def write_txt(out_path, names, labels, preds, probs_fall,
              metrics, threshold, weights_path, cfg):
    lines = []
    m = metrics

    lines.append("# Fall Detection NTU RGB+D 25-Joint — Hasil Test")
    lines.append("# Tanggal     : {}".format(
        datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    lines.append("# Model       : {}".format(weights_path))
    lines.append("# Config      : {}".format(cfg.get('work_dir', '-')))
    lines.append("# Threshold   : {:.2f}".format(threshold))
    lines.append("# num_point   : {}  (NTU RGB+D 25-joint)".format(
        cfg['model_args']['num_point']))
    lines.append("# Feeder      : {}".format(cfg['feeder']))
    lines.append("# Kelas FALL  : A043")
    lines.append("# Kelas NOT_FALL: A008, A009, A027, A042")
    lines.append("# " + "-" * 80)

    W = 45
    lines.append("# {:<{w}} | {:<10} | {:<10} | {:<7} | score_fall".format(
        "nama_sampel", "label_asli", "prediksi", "hasil", w=W))
    lines.append("# " + "-" * 80)

    for name, label, pred, score in zip(names, labels, preds, probs_fall):
        hasil = "BENAR" if label == pred else "SALAH"
        lines.append("{:<{w}} | {:<10} | {:<10} | {:<7} | {:.4f}".format(
            name,
            LABEL_NAMES[label],
            LABEL_NAMES[pred],
            hasil,
            score,
            w=W,
        ))

    tf = m['TP'] + m['FN']
    tn_total = m['TN'] + m['FP']

    lines.append("")
    lines.append("# " + "=" * 80)
    lines.append("# RINGKASAN METRIK")
    lines.append("# " + "=" * 80)
    lines.append("# Total sampel      : {}".format(m['n']))
    lines.append("# Distribusi        : not_fall={} fall={}".format(
        tn_total, tf))
    lines.append("# " + "-" * 50)
    lines.append("# Accuracy          : {:.4f}  ({:.2f}%)".format(
        m['acc'], m['acc'] * 100))
    lines.append("# Balanced Accuracy : {:.4f}  ({:.2f}%)".format(
        m['bal_acc'], m['bal_acc'] * 100))
    lines.append("# " + "-" * 50)
    lines.append("# Precision         : {:.4f}  ({:.2f}%)".format(
        m['prec'], m['prec'] * 100))
    lines.append("# Sensitivity/Recall: {:.4f}  ({:.2f}%)  <- penting untuk safety".format(
        m['sensitivity'], m['sensitivity'] * 100))
    lines.append("# Specificity       : {:.4f}  ({:.2f}%)".format(
        m['specificity'], m['specificity'] * 100))
    lines.append("# F1-Score          : {:.4f}  ({:.2f}%)".format(
        m['f1'], m['f1'] * 100))
    lines.append("# AUC-ROC           : {:.4f}".format(m['auc']))
    lines.append("# " + "-" * 50)
    lines.append("# Confusion Matrix:")
    lines.append("#                    Pred NOT_FALL  Pred FALL")
    lines.append("#   True NOT_FALL        {:^5}         {:^5}  (TN / FP)".format(
        m['TN'], m['FP']))
    lines.append("#   True FALL            {:^5}         {:^5}  (FN / TP)".format(
        m['FN'], m['TP']))
    if tf > 0:
        lines.append("# " + "-" * 50)
        lines.append("# Missed fall (FN): {}/{} = {:.1f}%".format(
            m['FN'], tf, m['FN'] / tf * 100))
    if tn_total > 0:
        lines.append("# False alarm (FP): {}/{} = {:.1f}%".format(
            m['FP'], tn_total, m['FP'] / tn_total * 100))

    content = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    return content


def threshold_sweep(labels, probs_fall):
    print("\n  Threshold sweep:")
    print("  {:>5}  {:>7}  {:>8}  {:>7}  {:>8}  {:>9}  {:>7}".format(
        "Thr", "Acc", "Bal_Acc", "Recall", "Specif", "F1", "AUC"))
    for thr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        preds = [1 if p >= thr else 0 for p in probs_fall]
        m = compute_metrics(labels, preds, probs_fall)
        mark = " <-" if abs(thr - 0.5) < 0.01 else ""
        print("  {:>5.1f}  {:>6.1f}%  {:>7.1f}%  {:>6.1f}%  {:>7.1f}%  {:>8.1f}%  {:>.4f}{}".format(
            thr,
            m['acc'] * 100,
            m['bal_acc'] * 100,
            m['sensitivity'] * 100,
            m['specificity'] * 100,
            m['f1'] * 100,
            m['auc'],
            mark,
        ))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Test model BlockGCN pada dataset NTU RGB+D 25-joint")
    ap.add_argument("--weights",    default=DEFAULT_WEIGHTS,
                    help="Path model weights .pt")
    ap.add_argument("--config",     default=DEFAULT_CONFIG,
                    help="Path config YAML (fall-detection-ntu)")
    ap.add_argument("--data",       default=DEFAULT_DATA,
                    help="Path val_data.npy (override config)")
    ap.add_argument("--labels",     default=DEFAULT_LABELS,
                    help="Path val_label.pkl (override config)")
    ap.add_argument("--out",        default=DEFAULT_OUT,
                    help="Path output .txt")
    ap.add_argument("--threshold",  type=float, default=0.5,
                    help="Threshold P(fall) untuk prediksi (default 0.5)")
    ap.add_argument("--batch_size", type=int,   default=16)
    ap.add_argument("--device",     default="cuda:0")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA tidak tersedia, pakai CPU")
        args.device = "cpu"

    print("\n" + "=" * 60)
    print("  NTU RGB+D 25-Joint — Fall Detection Test")
    print("=" * 60)
    print("  Model    : {}".format(args.weights))
    print("  Config   : {}".format(args.config))
    print("  Data     : {}".format(args.data))
    print("  Threshold: {}".format(args.threshold))
    print("  Output   : {}".format(args.out))

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── 1. Load model ──────────────────────────────────────────────────────────
    print("\nMemuat model ...")
    model = load_model(cfg, args.weights, args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print("  OK  ({:,} parameter)".format(n_params))

    # ── 2. Load dataset via Feeder ─────────────────────────────────────────────
    print("\nMemuat dataset via Feeder ...")
    dataset = load_dataset(cfg, args.data, args.labels)
    dist = dataset.class_distribution()
    print("  {} sampel  (not_fall={}, fall={})".format(
        dist['total'], dist['not_fall'], dist['fall']))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    # ── 3. Inferensi ───────────────────────────────────────────────────────────
    print("\nMenjalankan inferensi ...")
    all_labels = []
    all_preds  = []
    all_probs  = []

    for batch_idx, (data, label) in enumerate(loader):
        data = data.float().to(args.device)
        with torch.no_grad():
            output = model(data)
            probs  = torch.softmax(output, dim=1)
            pred   = torch.argmax(probs, dim=1)

        all_labels.extend(label.numpy().tolist())
        all_preds.extend(pred.cpu().numpy().tolist())
        all_probs.extend(probs[:, 1].cpu().numpy().tolist())

        done = min((batch_idx + 1) * args.batch_size, len(dataset))
        print("\r  {}/{}".format(done, len(dataset)), end="", flush=True)

    print()

    # ── 4. Hitung metrik ───────────────────────────────────────────────────────
    probs_fall = all_probs
    preds      = [1 if p >= args.threshold else 0 for p in probs_fall]
    metrics    = compute_metrics(all_labels, preds, probs_fall)

    # ── 5. Tulis .txt ──────────────────────────────────────────────────────────
    names = dataset.sample_name
    print("\nMenulis ke: {}".format(args.out))
    write_txt(
        out_path     = args.out,
        names        = names,
        labels       = all_labels,
        preds        = preds,
        probs_fall   = probs_fall,
        metrics      = metrics,
        threshold    = args.threshold,
        weights_path = args.weights,
        cfg          = cfg,
    )

    # ── 6. Ringkasan terminal ──────────────────────────────────────────────────
    m = metrics
    tf       = m['TP'] + m['FN']
    tn_total = m['TN'] + m['FP']

    print("\n" + "=" * 60)
    print("  HASIL TEST — NTU RGB+D 25-Joint")
    print("=" * 60)
    print("  Total sampel      : {}  (not_fall={} fall={})".format(
        m['n'], tn_total, tf))
    print("  " + "-" * 45)
    print("  Accuracy          : {:.4f}  ({:.2f}%)".format(
        m['acc'], m['acc'] * 100))
    print("  Balanced Accuracy : {:.4f}  ({:.2f}%)".format(
        m['bal_acc'], m['bal_acc'] * 100))
    print("  " + "-" * 45)
    print("  Precision         : {:.4f}  ({:.2f}%)".format(
        m['prec'], m['prec'] * 100))
    print("  Sensitivity/Recall: {:.4f}  ({:.2f}%)".format(
        m['sensitivity'], m['sensitivity'] * 100))
    print("  Specificity       : {:.4f}  ({:.2f}%)".format(
        m['specificity'], m['specificity'] * 100))
    print("  F1-Score          : {:.4f}  ({:.2f}%)".format(
        m['f1'], m['f1'] * 100))
    print("  AUC-ROC           : {:.4f}".format(m['auc']))
    print("\n  Confusion Matrix:")
    print("               Pred NOT_FALL  Pred FALL")
    print("  True NOT_FALL     {:^5}         {:^5}".format(m['TN'], m['FP']))
    print("  True FALL         {:^5}         {:^5}".format(m['FN'], m['TP']))
    if tf > 0:
        print("\n  Missed fall (FN): {}/{} = {:.1f}%".format(
            m['FN'], tf, m['FN'] / tf * 100))
    if tn_total > 0:
        print("  False alarm (FP): {}/{} = {:.1f}%".format(
            m['FP'], tn_total, m['FP'] / tn_total * 100))
    print("\n  Tersimpan di: {}".format(args.out))

    # ── 7. Threshold sweep ─────────────────────────────────────────────────────
    threshold_sweep(all_labels, probs_fall)
    print()


if __name__ == "__main__":
    main()
