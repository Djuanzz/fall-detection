#!/usr/bin/env python
"""
Optuna hyperparameter search untuk BlockGCN fall detection.
Target metric: best balanced accuracy pada val set.

Usage (jalankan dari root project):
    python scripts/tuning/hyperparameter_search.py
    python scripts/tuning/hyperparameter_search.py --n-trials 20 --search-epochs 30
    python scripts/tuning/hyperparameter_search.py --config config/fall-detection-yolo/balanced.yaml
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_trial(params: dict, base_config_path: str, search_epochs: int, trial_id: int) -> float:
    with open(base_config_path) as f:
        cfg = yaml.safe_load(f)

    cfg['base_lr'] = params['base_lr']
    cfg['weight_decay'] = params['weight_decay']
    cfg['model_args']['drop_out'] = params['drop_out']
    cfg['batch_size'] = params['batch_size']
    cfg['test_batch_size'] = params['batch_size']
    cfg['warm_up_epoch'] = params['warm_up_epoch']
    cfg['num_epoch'] = search_epochs
    cfg['save_epoch'] = search_epochs + 1  # no checkpoint saves during search
    cfg['save_score'] = False
    cfg['print_log'] = False

    step1 = max(1, int(search_epochs * params['step1_ratio']))
    step2 = max(step1 + 1, int(search_epochs * params['step2_ratio']))
    cfg['step'] = [step1, step2]

    trial_dir = os.path.join(BASE_DIR, 'work_dir', f'hpsearch_trial_{trial_id:03d}')
    shutil.rmtree(trial_dir, ignore_errors=True)
    cfg['work_dir'] = trial_dir

    tmp_fd, tmp_config = tempfile.mkstemp(suffix='.yaml')
    try:
        with os.fdopen(tmp_fd, 'w') as f:
            yaml.dump(cfg, f)

        # Paksa subprocess (main.py) memakai UTF-8 untuk stdout/stderr.
        # Tanpa ini, di Windows pipe yang ditangkap memakai cp1252 dan main.py
        # crash saat mencetak karakter non-ASCII (mis. '->'), sehingga setiap
        # trial mati sebelum training dan balanced accuracy selalu 0.
        child_env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
        result = subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, 'main.py'), '--config', tmp_config],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=BASE_DIR,
            timeout=7200,
            env=child_env,
        )
        output = result.stdout + result.stderr
    finally:
        os.unlink(tmp_config)
        shutil.rmtree(trial_dir, ignore_errors=True)

    bal_accs = re.findall(r'Balanced Accuracy: (\d+\.\d+)%', output)
    if not bal_accs:
        print(f'  [Trial {trial_id}] GAGAL (rc={result.returncode}) — tidak ada '
              f'balanced accuracy di output. Trial di-prune (tidak dicatat sebagai 0%).')
        if 'out of memory' in output:
            print('  -> Penyebab: CUDA OUT OF MEMORY. Kecilkan batch_size atau pakai GPU lebih besar.')
        print(f'  stderr tail:\n{output[-1200:]}')
        return None

    values = [float(x) / 100.0 for x in bal_accs]
    # Mean of last 5 epochs reduces noise
    tail = values[-5:]
    return float(sum(tail) / len(tail))


def objective(trial, base_config, search_epochs):
    try:
        import optuna
    except ImportError:
        raise SystemExit('optuna not installed. Run: pip install optuna')

    params = {
        'base_lr':       trial.suggest_float('base_lr',       1e-3, 5e-2,  log=True),
        'weight_decay':  trial.suggest_float('weight_decay',  1e-4, 5e-3,  log=True),
        'drop_out':      trial.suggest_categorical('drop_out',  [0.1, 0.2, 0.3, 0.5]),
        'batch_size':    trial.suggest_categorical('batch_size', [16, 32, 64]),
        'warm_up_epoch': trial.suggest_int('warm_up_epoch', 3, 10),
        'step1_ratio':   trial.suggest_float('step1_ratio', 0.50, 0.70),
        'step2_ratio':   trial.suggest_float('step2_ratio', 0.75, 0.90),
    }

    print(f'\n[Trial {trial.number}] '
          f'lr={params["base_lr"]:.5f}  wd={params["weight_decay"]:.5f}  '
          f'dropout={params["drop_out"]}  bs={params["batch_size"]}  '
          f'warmup={params["warm_up_epoch"]}  '
          f'step=[{params["step1_ratio"]:.2f}, {params["step2_ratio"]:.2f}]')

    score = run_trial(params, base_config, search_epochs, trial.number)
    if score is None:
        raise optuna.TrialPruned()
    print(f'[Trial {trial.number}] Balanced Acc: {score:.4f} ({score*100:.2f}%)')
    return score


def save_best_config(best_params: dict, base_config_path: str, full_epochs: int = 80) -> str:
    with open(base_config_path) as f:
        cfg = yaml.safe_load(f)

    cfg['base_lr'] = best_params['base_lr']
    cfg['weight_decay'] = best_params['weight_decay']
    cfg['model_args']['drop_out'] = best_params['drop_out']
    cfg['batch_size'] = best_params['batch_size']
    cfg['test_batch_size'] = best_params['batch_size']
    cfg['warm_up_epoch'] = best_params['warm_up_epoch']
    cfg['num_epoch'] = full_epochs
    cfg['save_epoch'] = 30
    cfg['save_score'] = True
    cfg['print_log'] = True

    step1 = max(1, int(full_epochs * best_params['step1_ratio']))
    step2 = max(step1 + 1, int(full_epochs * best_params['step2_ratio']))
    cfg['step'] = [step1, step2]

    # Nama output diturunkan dari config dasar: balanced.yaml -> balanced_best, full.yaml -> full_best
    stem = os.path.splitext(os.path.basename(base_config_path))[0]
    cfg['work_dir'] = f'work_dir/fall_yolo17_{stem}_best'

    out_path = os.path.join(
        os.path.dirname(base_config_path),
        f'{stem}_best.yaml',
    )
    with open(out_path, 'w') as f:
        yaml.dump(cfg, f)

    return out_path


def main():
    parser = argparse.ArgumentParser(description='Optuna hyperparameter search — BlockGCN')
    parser.add_argument('--config',         default='config/fall-detection-yolo/balanced.yaml')
    parser.add_argument('--n-trials',       type=int, default=20)
    parser.add_argument('--search-epochs',  type=int, default=30,
                        help='Epochs per trial (default 30). Shorter = faster search, more noise.')
    parser.add_argument('--study-name',     default=None)
    parser.add_argument('--full-epochs',    type=int, default=80,
                        help='num_epoch in saved best config')
    args = parser.parse_args()

    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        raise SystemExit('optuna not installed. Run: pip install optuna')

    base_config = os.path.join(BASE_DIR, args.config)
    if not os.path.exists(base_config):
        raise SystemExit(f'Config not found: {base_config}')

    study_name = args.study_name or f'hpsearch_{datetime.now():%Y%m%d_%H%M%S}'
    os.makedirs(os.path.join(BASE_DIR, 'work_dir'), exist_ok=True)
    db_path = os.path.join(BASE_DIR, 'work_dir', f'optuna_{study_name}.db')

    print(f'Study: {study_name}')
    print(f'Config: {base_config}')
    print(f'Trials: {args.n_trials}  Epochs/trial: {args.search_epochs}')
    print(f'DB: {db_path}')
    print('-' * 60)

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name=study_name,
        storage=f'sqlite:///{db_path}',
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, base_config, args.search_epochs),
        n_trials=args.n_trials,
        show_progress_bar=False,
    )

    best = study.best_trial
    print('\n' + '=' * 60)
    print(f'Best Trial: #{best.number}')
    print(f'  Balanced Accuracy: {best.value*100:.2f}%')
    for k, v in best.params.items():
        print(f'  {k}: {v}')

    out_path = save_best_config(best.params, base_config, args.full_epochs)
    print(f'\nBest config → {out_path}')
    print(f'Train full: python main.py --config {out_path} --device 0')

    completed = [t for t in study.trials if t.value is not None]
    top5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
    print('\nTop 5 trials:')
    for t in top5:
        p = t.params
        print(f'  #{t.number}: {t.value*100:.2f}%  '
              f'lr={p.get("base_lr"):.5f}  wd={p.get("weight_decay"):.5f}  '
              f'dropout={p.get("drop_out")}  bs={p.get("batch_size")}')


if __name__ == '__main__':
    main()
