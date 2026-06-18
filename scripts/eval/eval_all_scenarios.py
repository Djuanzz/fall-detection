"""
eval_all_scenarios.py
=====================
Re-eval OFFLINE checkpoint .pt untuk 4 skenario (17/25 x bal/imbal) sekaligus,
output tabel perbandingan + confusion matrix. Angka di sini = sumber resmi
(checkpoint dievaluasi ulang di test set), bukan baris eval di log.txt.

Set eksperimen dipilih via --set (default 03 = hasil akhir skripsi). Checkpoint
.pt tiap folder weights_new/<set>_<key>/ dideteksi otomatis (runs-*.pt).

Cara pakai:
    python scripts/eval_all_scenarios.py                 # set 03
    python scripts/eval_all_scenarios.py --set 04
    python scripts/eval_all_scenarios.py --set 04 --out docs/eval/hasil_04.txt

Selain tabel .txt, ditulis juga JSON confusion (docs/eval/confusion_<set>.json)
yang dibaca scripts/fig_paper_confusion.py untuk menggambar confusion matrix.
"""

import argparse
import importlib
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# ── Data test per konfigurasi (sama untuk semua set; yang beda hanya weights) ───
CONFIGS = {
    "17bal": {"name": "17-joint Balanced",
              "data": "dataset/yolo17_data/balanced/joint/val_data.npy",
              "labels": "dataset/yolo17_data/balanced/joint/val_label.pkl"},
    "17ful": {"name": "17-joint Imbalanced",
              "data": "dataset/yolo17_data/full/joint/val_data.npy",
              "labels": "dataset/yolo17_data/full/joint/val_label.pkl"},
    "25bal": {"name": "25-joint Balanced",
              "data": "dataset/ntu25_data/balanced/joint/val_data.npy",
              "labels": "dataset/ntu25_data/balanced/joint/val_label.pkl"},
    "25ful": {"name": "25-joint Imbalanced",
              "data": "dataset/ntu25_data/full/joint/val_data.npy",
              "labels": "dataset/ntu25_data/full/joint/val_label.pkl"},
}


def build_scenarios(set_name):
    """Susun 4 skenario untuk satu set; checkpoint .pt dideteksi otomatis."""
    scenarios = []
    for key, info in CONFIGS.items():
        folder = ROOT / "weights_new" / f"{set_name}_{key}"
        ckpts = sorted(folder.glob("runs-*.pt"))
        scenarios.append({
            "key":     key,
            "name":    info["name"],
            "weights": str(ckpts[0]) if ckpts else str(folder / "MISSING.pt"),
            "config":  str(folder / "config.yaml"),
            "data":    info["data"],
            "labels":  info["labels"],
        })
    return scenarios


# ── Utilities ──────────────────────────────────────────────────────────────────

def import_class(name):
    parts = name.split(".")
    mod   = importlib.import_module(".".join(parts[:-1]))
    return getattr(mod, parts[-1])


def load_model(cfg, weights_path, device):
    Model   = import_class(cfg["model"])
    model   = Model(**cfg["model_args"])
    weights = torch.load(weights_path, map_location="cpu")
    weights = {k.replace("module.", ""): v for k, v in weights.items()}
    model.load_state_dict(weights)
    return model.to(device).eval()


def load_dataset(cfg):
    Feeder      = import_class(cfg["feeder"])
    feeder_args = dict(cfg["test_feeder_args"])
    for k in ["random_move", "random_shift", "random_flip",
              "random_speed", "random_noise"]:
        feeder_args[k] = False
    feeder_args["split"]    = "val"
    feeder_args["use_mmap"] = False
    return Feeder(**feeder_args)


def compute_metrics(labels, preds, probs_fall):
    TP = sum(l == 1 and p == 1 for l, p in zip(labels, preds))
    TN = sum(l == 0 and p == 0 for l, p in zip(labels, preds))
    FP = sum(l == 0 and p == 1 for l, p in zip(labels, preds))
    FN = sum(l == 1 and p == 0 for l, p in zip(labels, preds))
    n   = len(labels)
    acc  = (TP + TN) / n
    prec = TP / max(TP + FP, 1)
    sens = TP / max(TP + FN, 1)
    spec = TN / max(TN + FP, 1)
    f1   = 2 * prec * sens / max(prec + sens, 1e-9)
    bal  = (sens + spec) / 2.0
    auc  = _auc(labels, probs_fall)
    return dict(TP=TP, TN=TN, FP=FP, FN=FN, n=n,
                acc=acc, bal_acc=bal, prec=prec,
                sens=sens, spec=spec, f1=f1, auc=auc)


def _auc(labels, scores):
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos = sum(labels); n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp, fp, prev, pts = 0, 0, None, [(0.0, 0.0)]
    for score, label in pairs:
        if prev is not None and score != prev:
            pts.append((fp / n_neg, tp / n_pos))
        if label == 1: tp += 1
        else:          fp += 1
        prev = score
    pts.append((1.0, 1.0))
    return sum((pts[i][0]-pts[i-1][0])*(pts[i-1][1]+pts[i][1])/2
               for i in range(1, len(pts)))


def run_inference(model, dataset, device, batch_size=32):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, num_workers=0, drop_last=False)
    all_labels, all_probs = [], []
    for data, label in loader:
        with torch.no_grad():
            probs = torch.softmax(model(data.float().to(device)), dim=1)
        all_labels.extend(label.numpy().tolist())
        all_probs.extend(probs[:, 1].cpu().numpy().tolist())
    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    return all_labels, preds, all_probs


# ── Report builder (format: hasil_evaluasi_p.txt) ───────────────────────────────

def build_report(results, model_name, threshold=0.5):
    lines = []
    lines.append(f"# Hasil Evaluasi {model_name} — Semua Skenario")
    lines.append(f"# Tanggal: {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"# Threshold: {threshold:.2f}")
    lines.append("")

    # ── Tabel ringkasan ────────────────────────────────────────────────────────
    header = (f"{'Skenario':<22}{'Acc':>8}{'BalAcc':>8}{'Prec':>8}"
              f"{'Sens':>8}{'Spec':>8}{'F1':>8}{'AUC':>8}{'FN':>5}{'FP':>5}")
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        m = r["m"]
        lines.append(
            f"{r['name']:<22}"
            f"{m['acc']*100:>7.2f}% {m['bal_acc']*100:>6.2f}% "
            f"{m['prec']*100:>6.2f}% {m['sens']*100:>6.2f}% "
            f"{m['spec']*100:>6.2f}% {m['f1']*100:>6.2f}% "
            f"{m['auc']:>7.4f}{m['FN']:>5}{m['FP']:>5}")
    lines.append("")

    # ── Confusion matrix ───────────────────────────────────────────────────────
    lines.append("-" * len(header))
    lines.append("CONFUSION MATRIX")
    lines.append("-" * len(header))
    lines.append("")
    for r in results:
        m = r["m"]
        lines.append(f"{r['name']}:")
        lines.append(f"{'':18}Pred NOT_FALL  Pred FALL")
        lines.append(f"{'  True NOT_FALL':<18}{m['TN']:^13}  {m['FP']:^9}")
        lines.append(f"{'  True FALL':<18}{m['FN']:^13}  {m['TP']:^9}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set",     default="03",
                    help="set eksperimen: 01/02/03/04 (folder weights_new/<set>_*)")
    ap.add_argument("--out",     default=None,
                    help="path tabel .txt (default docs/eval/hasil_<set>.txt)")
    ap.add_argument("--json",    default=None,
                    help="path JSON confusion (default docs/eval/confusion_<set>.json)")
    ap.add_argument("--device",  default="cuda:0")
    ap.add_argument("--batch",   type=int, default=32)
    args = ap.parse_args()

    out_txt  = ROOT / (args.out  or f"docs/eval/hasil_{args.set}.txt")
    out_json = ROOT / (args.json or f"docs/eval/confusion_{args.set}.json")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA tidak tersedia, pakai CPU")
        args.device = "cpu"

    results = []
    confusion = {}  # key -> {TN, FP, FN, TP}

    for sc in build_scenarios(args.set):
        print(f"\n{'='*55}")
        print(f"  [{args.set}] {sc['name']}")
        print(f"{'='*55}")

        cfg_path = Path(sc["config"])
        wgt_path = Path(sc["weights"])

        if not cfg_path.exists():
            print(f"  [SKIP] Config tidak ditemukan: {cfg_path}")
            continue
        if not wgt_path.exists():
            print(f"  [SKIP] Weights tidak ditemukan: {wgt_path}")
            continue

        print(f"  Checkpoint: {wgt_path.name}")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        # Override data path ke yang spesifik di CONFIGS
        cfg["test_feeder_args"]["data_path"]  = str(ROOT / sc["data"])
        cfg["test_feeder_args"]["label_path"] = str(ROOT / sc["labels"])

        print("  Memuat model ...", end=" ", flush=True)
        model = load_model(cfg, str(wgt_path), args.device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"OK ({n_params:,} params)")

        print("  Memuat dataset ...", end=" ", flush=True)
        dataset = load_dataset(cfg)
        print(f"OK ({len(dataset)} sampel)")

        print("  Inferensi ...", end=" ", flush=True)
        labels, preds, probs = run_inference(model, dataset, args.device, args.batch)
        m = compute_metrics(labels, preds, probs)
        print("OK")

        results.append({"name": sc["name"], "m": m, "n_params": n_params,
                         "model": cfg["model"].split(".")[-2]})
        confusion[sc["key"]] = {"TN": m["TN"], "FP": m["FP"],
                                "FN": m["FN"], "TP": m["TP"],
                                "checkpoint": wgt_path.name}

    if not results:
        print("\n[ERROR] Tidak ada skenario yang berhasil dievaluasi.")
        return

    model_name = results[0]["model"]
    report = build_report(results, model_name, threshold=0.5)

    # ── Cetak ke konsol ─────────────────────────────────────────────────────────
    print(f"\n{'='*len(report.splitlines()[4])}")
    print(report)

    # ── Simpan tabel .txt ───────────────────────────────────────────────────────
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Disimpan tabel : {out_txt}")

    # ── Simpan JSON confusion (dibaca fig_paper_confusion.py) ────────────────────
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(confusion, f, indent=2)
    print(f"Disimpan JSON  : {out_json}")


if __name__ == "__main__":
    main()
