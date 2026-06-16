#!/usr/bin/env python
"""
Hyperparameter search BlockGCN — VARIAN COLAB (progres disimpan ke Google Drive).

Beda dengan scripts/tuning/hyperparameter_search.py:
  * Semua progres (DB Optuna, config terbaik, ringkasan) disalin ke Google Drive,
    sehingga AMAN dari disconnect Colab — tinggal jalankan ulang perintah yang
    sama untuk MELANJUTKAN (resume) trial yang sudah ada, tidak mulai dari nol.
  * DB Optuna ditaruh di disk lokal Colab yang cepat saat berjalan, lalu disalin
    ke Drive setiap selesai 1 trial (menghindari masalah file-lock SQLite di
    Google Drive FUSE). Saat start, DB dipulihkan dari Drive bila ada.
  * Logika trial (termasuk fix encoding + prune trial gagal) DIPAKAI ULANG dari
    hyperparameter_search.py, tidak diduplikasi.

Cara pakai di Colab (jalankan dari root project):
    !pip install optuna
    # mount Drive sekali:
    from google.colab import drive; drive.mount('/content/drive')
    # lalu:
    !python scripts/tuning/hyperparameter_search_colab.py \
        --config config/fall-detection-yolo/balanced.yaml --n-trials 60

Hasil untuk config 'balanced.yaml' (stem=balanced) tersimpan di:
    /content/drive/MyDrive/Fall-Detection/work_dir/fall_yolo17_balanced/
        optuna_<study>.db      <- database Optuna (untuk resume)
        balanced_best.yaml     <- config terbaik (siap untuk training penuh)
        search_summary.txt     <- ringkasan: best + top 10, diperbarui tiap trial
Jalankan ulang perintah yang sama untuk menambah / melanjutkan trial.
"""

import argparse
import os
import shutil
import sys
from datetime import datetime

import yaml

# Pakai ulang fungsi dari script search utama (sudah berisi fix encoding + prune).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hyperparameter_search as hps  # noqa: E402

DEFAULT_DRIVE_DIR = '/content/drive/MyDrive/Fall-Detection/param'


def write_summary(study, optuna, path, study_name, config, search_epochs):
    lines = [
        f'Study        : {study_name}',
        f'Config       : {config}',
        f'Epochs/trial : {search_epochs}',
        f'Diperbarui   : {datetime.now():%Y-%m-%d %H:%M:%S}',
    ]
    completed = [t for t in study.trials if t.value is not None]
    n_pruned = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.PRUNED])
    lines.append(f'Total trial  : {len(study.trials)}  '
                 f'(selesai: {len(completed)}, prune/gagal: {n_pruned})')
    if completed:
        b = study.best_trial
        lines += ['', f'BEST #{b.number}: Balanced Accuracy {b.value*100:.2f}%']
        lines += [f'  {k}: {v}' for k, v in b.params.items()]
        lines += ['', 'Top 10 trial:']
        for t in sorted(completed, key=lambda t: t.value, reverse=True)[:10]:
            p = t.params
            lines.append(
                f'  #{t.number}: {t.value*100:.2f}%  '
                f'lr={p["base_lr"]:.5f}  wd={p["weight_decay"]:.5f}  '
                f'dropout={p["drop_out"]}  bs={p["batch_size"]}  '
                f'warmup={p["warm_up_epoch"]}  '
                f'step=[{p["step1_ratio"]:.2f}, {p["step2_ratio"]:.2f}]')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Optuna hyperparameter search BlockGCN - varian Colab (simpan ke Drive)')
    parser.add_argument('--config',        default='config/fall-detection-yolo/balanced.yaml')
    parser.add_argument('--n-trials',      type=int, default=60)
    parser.add_argument('--search-epochs', type=int, default=30,
                        help='Epoch per trial (default 30). Lebih kecil = cepat tapi lebih noisy.')
    parser.add_argument('--drive-dir',     default=DEFAULT_DRIVE_DIR,
                        help='Folder dasar di Google Drive untuk menyimpan hasil.')
    parser.add_argument('--study-name',    default=None,
                        help='Default: hpsearch_<stem> (mis. hpsearch_balanced) agar resume otomatis.')
    parser.add_argument('--full-epochs',   type=int, default=80,
                        help='num_epoch pada config terbaik yang disimpan.')
    args = parser.parse_args()

    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        raise SystemExit('optuna belum terpasang. Jalankan: pip install optuna')

    base_config = os.path.join(hps.BASE_DIR, args.config)
    if not os.path.exists(base_config):
        raise SystemExit(f'Config tidak ditemukan: {base_config}')

    stem = os.path.splitext(os.path.basename(base_config))[0]          # balanced / full
    study_name = args.study_name or f'hpsearch_{stem}'

    # Cek Drive ter-mount (hanya jika memang menargetkan /content/drive).
    if args.drive_dir.startswith('/content/drive') and not os.path.isdir('/content/drive/MyDrive'):
        raise SystemExit(
            "Google Drive belum di-mount. Jalankan dulu di sel Colab:\n"
            "    from google.colab import drive\n"
            "    drive.mount('/content/drive')")

    out_dir = os.path.join(args.drive_dir, f'fall_yolo17_{stem}')
    os.makedirs(out_dir, exist_ok=True)

    # DB lokal (cepat) saat berjalan; salinan persisten di Drive.
    os.makedirs(os.path.join(hps.BASE_DIR, 'work_dir'), exist_ok=True)
    local_db = os.path.join(hps.BASE_DIR, 'work_dir', f'optuna_{study_name}.db')
    drive_db = os.path.join(out_dir, f'optuna_{study_name}.db')
    summary_path = os.path.join(out_dir, 'search_summary.txt')

    # Resume: pulihkan DB dari Drive bila ada dan belum ada salinan lokal.
    if os.path.exists(drive_db) and not os.path.exists(local_db):
        shutil.copy2(drive_db, local_db)
        print(f'[resume] DB dipulihkan dari Drive: {drive_db}')

    print(f'Study        : {study_name}')
    print(f'Config       : {base_config}')
    print(f'Trials       : {args.n_trials}   Epochs/trial: {args.search_epochs}')
    print(f'DB lokal     : {local_db}')
    print(f'Output Drive : {out_dir}')
    print('-' * 60)

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name=study_name,
        storage=f'sqlite:///{local_db}',
        load_if_exists=True,
    )

    done_before = len([t for t in study.trials if t.value is not None])
    if done_before:
        print(f'[resume] {len(study.trials)} trial sudah ada '
              f'({done_before} selesai). Melanjutkan...')

    def sync_to_drive(study, trial):
        """Dipanggil setelah TIAP trial: salin DB + tulis ringkasan ke Drive."""
        try:
            shutil.copy2(local_db, drive_db)
            write_summary(study, optuna, summary_path, study_name,
                          base_config, args.search_epochs)
            print(f'[drive-sync] progres tersimpan ke Drive (setelah trial #{trial.number}).')
        except Exception as e:  # noqa: BLE001 - jangan sampai gagal sync menghentikan search
            print(f'[drive-sync] PERINGATAN: gagal menyimpan ke Drive: {e}')

    study.optimize(
        lambda trial: hps.objective(trial, base_config, args.search_epochs),
        n_trials=args.n_trials,
        show_progress_bar=False,
        callbacks=[sync_to_drive],
    )

    completed = [t for t in study.trials if t.value is not None]
    if not completed:
        print('\nTidak ada trial yang berhasil (semua prune/gagal). '
              'Cek log di atas — kemungkinan CUDA OOM / config salah.')
        return

    best = study.best_trial
    print('\n' + '=' * 60)
    print(f'Best Trial: #{best.number}  Balanced Accuracy: {best.value*100:.2f}%')
    for k, v in best.params.items():
        print(f'  {k}: {v}')

    # Simpan config terbaik, lalu arahkan work_dir-nya ke Drive supaya training
    # penuh nanti juga tersimpan di Drive (sesuai contoh: .../fall_yolo17_<stem>).
    best_cfg_path = hps.save_best_config(best.params, base_config, args.full_epochs)
    with open(best_cfg_path) as f:
        bc = yaml.safe_load(f)
    bc['work_dir'] = out_dir
    with open(best_cfg_path, 'w') as f:
        yaml.dump(bc, f)
    drive_cfg = os.path.join(out_dir, os.path.basename(best_cfg_path))
    shutil.copy2(best_cfg_path, drive_cfg)

    # Sinkron terakhir + ringkasan final.
    try:
        shutil.copy2(local_db, drive_db)
        write_summary(study, optuna, summary_path, study_name,
                      base_config, args.search_epochs)
    except Exception as e:  # noqa: BLE001
        print(f'[drive-sync] PERINGATAN: gagal sinkron akhir: {e}')

    print(f'\nConfig terbaik   : {drive_cfg}')
    print(f'Ringkasan        : {summary_path}')
    print(f'DB (resume)      : {drive_db}')
    print(f'\nTraining penuh   : python main.py --config {best_cfg_path} --device 0')

    print('\nTop 5 trial:')
    for t in sorted(completed, key=lambda t: t.value, reverse=True)[:5]:
        p = t.params
        print(f'  #{t.number}: {t.value*100:.2f}%  '
              f'lr={p["base_lr"]:.5f}  wd={p["weight_decay"]:.5f}  '
              f'dropout={p["drop_out"]}  bs={p["batch_size"]}')


if __name__ == '__main__':
    main()
