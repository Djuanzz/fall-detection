"""
scripts/evaluate.py
====================
Evaluasi komprehensif model BlockGCN fall detection.

Metrics yang dihitung:
  - Accuracy, Balanced Accuracy
  - Precision, Recall (Sensitivity), Specificity
  - F1-Score, F2-Score (recall-weighted)
  - Matthews Correlation Coefficient (MCC)
  - AUC-ROC
  - Average Precision / PR-AUC
  - Confusion Matrix
  - Threshold sweep (optimal threshold via Youden index)
  - Per-sample hasil (CSV)

Output:
  - Terminal: ringkasan lengkap
  - <out_dir>/report.json     — semua metrik dalam JSON
  - <out_dir>/results.csv     — per-sample (name, true, pred, score_fall)
  - <out_dir>/roc_curve.png   — ROC curve (jika matplotlib tersedia)
  - <out_dir>/pr_curve.png    — Precision-Recall curve
  - <out_dir>/conf_matrix.png — Confusion matrix heatmap

Cara pakai (dari folder scripts/):
    python evaluate.py
    python evaluate.py --weights ../work_dir/fall_balanced/runs-XX-XXXX.pt
    python evaluate.py --weights ../weights/10/runs-23-3289.pt --threshold 0.4
"""

import argparse
import csv
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS  = "../weights/10/runs-23-3289.pt"
DEFAULT_CONFIG   = "../config/fall-detection/balanced.yaml"
DEFAULT_DATA     = "../dataset/ntu_data/balanced/joint/val_data.npy"
DEFAULT_LABELS   = "../dataset/ntu_data/balanced/joint/val_label.pkl"
DEFAULT_OUT_DIR  = "../work_dir/evaluation"

LABEL_NAMES = {0: "not_fall", 1: "fall"}


# ── Utilities ─────────────────────────────────────────────────────────────────

def import_class(name):
    parts = name.split('.')
    mod   = importlib.import_module('.'.join(parts[:-1]))
    return getattr(mod, parts[-1])


def load_model(cfg, weights_path, device):
    Model   = import_class(cfg["model"])
    model   = Model(**cfg["model_args"])
    weights = torch.load(weights_path, map_location="cpu", weights_only=False)
    weights = {k.replace("module.", ""): v for k, v in weights.items()}
    model.load_state_dict(weights)
    model.to(device).eval()
    return model


def load_dataset(cfg, data_path, labels_path):
    Feeder      = import_class(cfg["feeder"])
    feeder_args = dict(cfg["test_feeder_args"])

    if data_path:
        feeder_args["data_path"]  = data_path
    if labels_path:
        feeder_args["label_path"] = labels_path

    feeder_args.update({
        "random_move":  False,
        "random_shift": False,
        "random_flip":  False,
        "random_speed": False,
        "random_noise": False,
        "split":        "val",
        "use_mmap":     False,
    })
    return Feeder(**feeder_args)


# ── Metric computation ────────────────────────────────────────────────────────

def compute_metrics(labels, preds, probs_fall=None, threshold=0.5):
    labels = np.array(labels)
    preds  = np.array(preds)

    TP = int(((labels == 1) & (preds == 1)).sum())
    TN = int(((labels == 0) & (preds == 0)).sum())
    FP = int(((labels == 0) & (preds == 1)).sum())
    FN = int(((labels == 1) & (preds == 0)).sum())
    n  = len(labels)

    acc   = (TP + TN) / n
    prec  = TP / max(TP + FP, 1)
    rec   = TP / max(TP + FN, 1)     # Sensitivity / Recall
    spec  = TN / max(TN + FP, 1)     # Specificity
    npv   = TN / max(TN + FN, 1)     # Negative Predictive Value
    f1    = 2 * prec * rec / max(prec + rec, 1e-9)
    f2    = 5 * prec * rec / max(4 * prec + rec, 1e-9)  # beta=2, recall-weighted
    bal   = (rec + spec) / 2          # Balanced accuracy

    denom_mcc = ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) ** 0.5
    mcc   = (TP*TN - FP*FN) / max(denom_mcc, 1e-9)

    metrics = dict(
        TP=TP, TN=TN, FP=FP, FN=FN, n=n,
        accuracy=acc, balanced_accuracy=bal,
        precision=prec, recall=rec,
        specificity=spec, npv=npv,
        f1=f1, f2=f2, mcc=mcc,
        threshold=threshold,
        missed_fall_rate=FN / max(TP + FN, 1),
        false_alarm_rate=FP / max(TN + FP, 1),
    )

    if probs_fall is not None:
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            probs_fall = np.array(probs_fall)
            metrics['auc_roc'] = float(roc_auc_score(labels, probs_fall))
            metrics['pr_auc']  = float(average_precision_score(labels, probs_fall))
        except ImportError:
            pass
        except Exception:
            pass

    return metrics


def youden_threshold(labels, probs_fall):
    """Find threshold that maximises Youden index (Sensitivity + Specificity - 1)."""
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(labels, probs_fall)
        j     = tpr - fpr
        idx   = int(np.argmax(j))
        return float(thresholds[idx]), float(j[idx])
    except ImportError:
        return 0.5, 0.0


def threshold_sweep(labels, probs_fall, thresholds=None):
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5,
                      0.55, 0.6, 0.65, 0.7, 0.8, 0.9]
    labels = np.array(labels)
    rows   = []
    for thr in thresholds:
        preds = (np.array(probs_fall) >= thr).astype(int)
        TP = int(((labels == 1) & (preds == 1)).sum())
        TN = int(((labels == 0) & (preds == 0)).sum())
        FP = int(((labels == 0) & (preds == 1)).sum())
        FN = int(((labels == 1) & (preds == 0)).sum())
        n  = len(labels)
        prec = TP / max(TP + FP, 1)
        rec  = TP / max(TP + FN, 1)
        spec = TN / max(TN + FP, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        acc  = (TP + TN) / n
        miss = FN / max(TP + FN, 1)
        rows.append(dict(thr=thr, acc=acc, prec=prec, rec=rec,
                         spec=spec, f1=f1, miss=miss,
                         TP=TP, TN=TN, FP=FP, FN=FN))
    return rows


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_roc(labels, probs_fall, out_path, auc_val=None):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(labels, probs_fall)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, 'b-', lw=2,
                 label='ROC (AUC = {:.4f})'.format(auc_val or 0))
        plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Recall)')
        plt.title('ROC Curve — Fall Detection BlockGCN')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print("  ROC curve  → {}".format(out_path))
    except Exception as e:
        print("  [SKIP] ROC plot: {}".format(e))


def plot_pr(labels, probs_fall, out_path, pr_auc=None):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve

        prec, rec, _ = precision_recall_curve(labels, probs_fall)
        baseline = labels.count(1) / len(labels)
        plt.figure(figsize=(6, 5))
        plt.plot(rec, prec, 'r-', lw=2,
                 label='PR curve (AP = {:.4f})'.format(pr_auc or 0))
        plt.axhline(baseline, color='k', linestyle='--', lw=1,
                    label='Baseline (prevalence)')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve — Fall Detection BlockGCN')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print("  PR  curve  → {}".format(out_path))
    except Exception as e:
        print("  [SKIP] PR  plot: {}".format(e))


def plot_confusion(tn, fp, fn, tp, out_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        cm = np.array([[tn, fp], [fn, tp]])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap='Blues')
        plt.colorbar(im, ax=ax)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred NOT_FALL', 'Pred FALL'])
        ax.set_yticklabels(['True NOT_FALL', 'True FALL'])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black',
                        fontsize=18, fontweight='bold')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print("  Conf matrix → {}".format(out_path))
    except Exception as e:
        print("  [SKIP] Confusion matrix plot: {}".format(e))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Comprehensive evaluation for BlockGCN fall detection")
    ap.add_argument("--weights",    default=DEFAULT_WEIGHTS)
    ap.add_argument("--config",     default=DEFAULT_CONFIG)
    ap.add_argument("--data",       default=DEFAULT_DATA)
    ap.add_argument("--labels",     default=DEFAULT_LABELS)
    ap.add_argument("--out_dir",    default=DEFAULT_OUT_DIR)
    ap.add_argument("--threshold",  type=float, default=0.5,
                    help="Softmax threshold for fall (default 0.5; "
                         "use 'auto' via --threshold -1 for Youden optimum)")
    ap.add_argument("--batch_size", type=int,   default=16)
    ap.add_argument("--device",     default="cuda:0")
    ap.add_argument("--no_plots",   action="store_true",
                    help="Skip saving PNG plots")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA tidak tersedia, pakai CPU")
        args.device = "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  Fall Detection — Evaluasi Komprehensif (BlockGCN)")
    print("=" * 65)
    print("  Model      : {}".format(args.weights))
    print("  Config     : {}".format(args.config))
    print("  Data       : {}".format(args.data))
    print("  Output dir : {}".format(args.out_dir))

    # 1. Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # 2. Load model
    print("\n[1/4] Memuat model ...")
    model = load_model(cfg, args.weights, args.device)
    num_params = sum(p.numel() for p in model.parameters())
    print("  OK — {} parameter".format(num_params))

    # 3. Load dataset
    print("[2/4] Memuat dataset ...")
    dataset = load_dataset(cfg, args.data, args.labels)
    dist    = dataset.class_distribution()
    print("  {} sampel → not_fall={}, fall={}".format(
        dist['total'], dist['not_fall'], dist['fall']))

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0, drop_last=False)

    # 4. Inference
    print("[3/4] Inferensi ...")
    all_labels = []
    all_probs  = []

    for data, label in loader:
        data = data.float().to(args.device)
        with torch.no_grad():
            logits = model(data)
            probs  = torch.softmax(logits, dim=1)
        all_labels.extend(label.numpy().tolist())
        all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    all_labels   = list(all_labels)
    probs_fall   = list(all_probs)

    # 5. Find optimal threshold (Youden)
    opt_thr, youden_j = youden_threshold(all_labels, probs_fall)
    print("  Optimal threshold (Youden): {:.3f}  (J={:.4f})".format(
        opt_thr, youden_j))

    use_thr = args.threshold if args.threshold > 0 else opt_thr
    preds   = [1 if p >= use_thr else 0 for p in probs_fall]

    # 6. Compute metrics
    print("[4/4] Menghitung metrik ...")
    metrics = compute_metrics(all_labels, preds, probs_fall, use_thr)
    metrics['optimal_threshold_youden']  = opt_thr
    metrics['youden_j']                  = float(youden_j)
    metrics['num_params']                = num_params
    metrics['weights_path']              = args.weights
    metrics['timestamp']                 = datetime.now().isoformat()
    metrics['dataset_distribution']      = dist

    # 7. Save per-sample CSV
    csv_path = out_dir / "results.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'true_label', 'true_name',
                         'pred_label', 'pred_name', 'correct',
                         'score_fall', 'score_not_fall'])
        for i, (name, label, pred, prob) in enumerate(
                zip(dataset.sample_name, all_labels, preds, probs_fall)):
            writer.writerow([
                name, label, LABEL_NAMES[label],
                pred, LABEL_NAMES[pred],
                'BENAR' if label == pred else 'SALAH',
                '{:.6f}'.format(prob),
                '{:.6f}'.format(1 - prob),
            ])
    print("  CSV         → {}".format(csv_path))

    # 8. Save JSON report
    json_path = out_dir / "report.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print("  JSON report → {}".format(json_path))

    # 9. Plots
    if not args.no_plots:
        plot_roc(all_labels, probs_fall,
                 str(out_dir / "roc_curve.png"),
                 metrics.get('auc_roc'))
        plot_pr(all_labels, probs_fall,
                str(out_dir / "pr_curve.png"),
                metrics.get('pr_auc'))
        plot_confusion(metrics['TN'], metrics['FP'],
                       metrics['FN'], metrics['TP'],
                       str(out_dir / "conf_matrix.png"))

    # 10. Threshold sweep table
    sweep = threshold_sweep(all_labels, probs_fall)

    # 11. Print summary
    m = metrics
    print("\n" + "=" * 65)
    print("  HASIL EVALUASI")
    print("=" * 65)
    print("  Total sampel         : {}".format(m['n']))
    print("  Threshold digunakan  : {:.3f}".format(use_thr))
    print("  Threshold optimal    : {:.3f}  (Youden J={:.4f})".format(
        opt_thr, youden_j))
    print()
    print("  Accuracy             : {:.4f}  ({:.2f}%)".format(
        m['accuracy'], m['accuracy'] * 100))
    print("  Balanced Accuracy    : {:.4f}  ({:.2f}%)".format(
        m['balanced_accuracy'], m['balanced_accuracy'] * 100))
    print()
    print("  Precision            : {:.4f}  ({:.2f}%)".format(
        m['precision'], m['precision'] * 100))
    print("  Recall (Sensitivity) : {:.4f}  ({:.2f}%)  ← penting (jangan miss fall)".format(
        m['recall'], m['recall'] * 100))
    print("  Specificity          : {:.4f}  ({:.2f}%)".format(
        m['specificity'], m['specificity'] * 100))
    print("  NPV                  : {:.4f}  ({:.2f}%)".format(
        m['npv'], m['npv'] * 100))
    print()
    print("  F1-Score             : {:.4f}  ({:.2f}%)".format(
        m['f1'], m['f1'] * 100))
    print("  F2-Score             : {:.4f}  ({:.2f}%)".format(
        m['f2'], m['f2'] * 100))
    print("  MCC                  : {:.4f}".format(m['mcc']))
    if 'auc_roc' in m:
        print("  AUC-ROC              : {:.4f}".format(m['auc_roc']))
    if 'pr_auc' in m:
        print("  PR-AUC               : {:.4f}".format(m['pr_auc']))
    print()
    print("  Confusion Matrix:")
    print("                     Pred NOT_FALL   Pred FALL")
    print("  True NOT_FALL  {:^15}  {:^10}".format(m['TN'], m['FP']))
    print("  True FALL      {:^15}  {:^10}".format(m['FN'], m['TP']))
    print()
    print("  Missed fall    : {}/{} = {:.1f}%".format(
        m['FN'], m['TP'] + m['FN'],
        m['missed_fall_rate'] * 100))
    print("  False alarm    : {}/{} = {:.1f}%".format(
        m['FP'], m['TN'] + m['FP'],
        m['false_alarm_rate'] * 100))

    print("\n  ── Threshold Sweep ───────────────────────────────────────")
    print("  {:>5}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}".format(
        "Thr", "Acc", "Precision", "Recall", "Spec", "F1", "Missed"))
    for row in sweep:
        mark = " ◄" if abs(row['thr'] - use_thr) < 0.01 else \
               " ★" if abs(row['thr'] - opt_thr) < 0.01 else ""
        print("  {:>5.2f}  {:>7.2f}%  {:>8.2f}%  {:>7.2f}%  {:>7.2f}%  {:>7.2f}%  {:>7.2f}%{}".format(
            row['thr'],
            row['acc']  * 100,
            row['prec'] * 100,
            row['rec']  * 100,
            row['spec'] * 100,
            row['f1']   * 100,
            row['miss'] * 100,
            mark,
        ))
    print("  ◄ = threshold digunakan  ★ = optimal (Youden)")

    print("\n  Output tersimpan di: {}".format(out_dir))
    print()


if __name__ == "__main__":
    main()
