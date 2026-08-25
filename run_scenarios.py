"""
Thesis Benchmark Scenario Orchestrator: Maximum -> Medium -> Low
Date-Versioned & Session-Isolated Experiment Tracker

Directory Organization:
  experiments/
  ├── GLOBAL_ARCHIVE_INDEX.md               # Directory of all benchmark runs by date & session
  ├── 18_08_2026/                           # Previous session results (Preserved)
  │   ├── master_leaderboard.csv
  │   ├── SUMMARY.md
  │   └── scenarios/
  └── 19_08_2026_session1/                  # New session runs (Isolated)
      ├── master_leaderboard.csv
      ├── SUMMARY.md
      └── scenarios/
          ├── maximum/
          ├── medium/
          └── low/
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
import pandas as pd


def parse_args():
    default_date_str = datetime.now().strftime('%d_%m_%Y')
    parser = argparse.ArgumentParser(description="Thesis Benchmark Scenario Orchestrator with Date-Isolated Sessions")
    parser.add_argument('--session-id', type=str, default=None,
                        help=f"Session directory name inside experiments/ (default: current date, e.g. '{default_date_str}')")
    parser.add_argument('--resume', action='store_true', default=False,
                        help="Resume existing session instead of creating an incremented run folder (e.g. 18_08_2026_run2)")
    parser.add_argument('--scenario', type=str, default='all',
                        choices=['all', 'standard', 'medium', 'low', 'maximum'],
                        help="Scenario to run (default: all active scenarios: Standard -> Medium -> Low)")
    parser.add_argument('--models', nargs='+', default=None,
                        help="Subset of models to run (e.g. --models v1 v4conv v5). Default: all models in scenario.")
    parser.add_argument('--val-dataset', type=str, default='both',
                        choices=['ham10000', 'pad-ufes-20', 'both'], help="Validation dataset source")
    parser.add_argument('--scenarios-file', type=str, default='benchmark_scenarios.json',
                        help="Path to scenarios JSON config")
    parser.add_argument('--experiments-root', type=str, default='./experiments',
                        help="Root experiments directory")
    parser.add_argument('--python-bin', type=str, default=sys.executable,
                        help="Path to Python virtualenv binary")
    return parser.parse_args()


def resolve_session_directory(root_dir: Path, session_id: str = None, resume: bool = False) -> tuple:
    """Resolves session directory name. Auto-increments (run2, run3) if multiple runs occur on same day."""
    base_date = datetime.now().strftime('%d_%m_%Y')
    if session_id:
        session_name = session_id
        session_dir = root_dir / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_name, session_dir

    base_dir = root_dir / base_date
    if not base_dir.exists() or resume:
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_date, base_dir

    if (base_dir / 'master_leaderboard.csv').exists():
        run_idx = 2
        while (root_dir / f"{base_date}_run{run_idx}").exists() and (root_dir / f"{base_date}_run{run_idx}" / 'master_leaderboard.csv').exists():
            run_idx += 1
        session_name = f"{base_date}_run{run_idx}"
        session_dir = root_dir / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_name, session_dir
    else:
        return base_date, base_dir


def is_experiment_completed(exp_dir: Path) -> bool:
    """Check if experiment finished and produced valid results."""
    res_file = exp_dir / 'results.json'
    cfg_file = exp_dir / 'config.json'
    return res_file.exists() and cfg_file.exists() and res_file.stat().st_size > 10


def update_master_leaderboard(session_dir: Path, new_entry: dict = None):
    """Update session master leaderboard CSV and markdown summary table."""
    leaderboard_csv = session_dir / 'master_leaderboard.csv'
    if new_entry is not None:
        if leaderboard_csv.exists():
            df = pd.read_csv(leaderboard_csv)
            df = df[df['experiment_id'] != new_entry['experiment_id']]
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        else:
            df = pd.DataFrame([new_entry])

        scenario_order = {'standard': 1, 'medium': 2, 'low': 3, 'maximum': 4}
        model_order = {'v1': 1, 'v2': 2, 'v3': 3, 'v3small': 3, 'v3large': 3, 'v4': 4, 'v4conv': 4, 'v4convl': 4, 'v5': 5}

        df['_scenario_rank'] = df['scenario'].map(lambda x: scenario_order.get(str(x).lower(), 99))
        df['_model_rank'] = df['model'].map(lambda x: model_order.get(str(x).lower(), 99))

        sort_cols = ['_scenario_rank', '_model_rank']
        ascending = [True, True]

        if 'ham_mel_recall' in df.columns:
            sort_cols.append('ham_mel_recall')
            ascending.append(False)
        if 'mel_recall' in df.columns:
            sort_cols.append('mel_recall')
            ascending.append(False)

        df.sort_values(by=sort_cols, ascending=ascending, inplace=True)
        df.drop(columns=['_scenario_rank', '_model_rank'], inplace=True, errors='ignore')
        df.to_csv(leaderboard_csv, index=False)
    elif leaderboard_csv.exists():
        df = pd.read_csv(leaderboard_csv)
        scenario_order = {'standard': 1, 'medium': 2, 'low': 3, 'maximum': 4}
        model_order = {'v1': 1, 'v2': 2, 'v3': 3, 'v3small': 3, 'v3large': 3, 'v4': 4, 'v4conv': 4, 'v4convl': 4, 'v5': 5}

        df['_scenario_rank'] = df['scenario'].map(lambda x: scenario_order.get(str(x).lower(), 99))
        df['_model_rank'] = df['model'].map(lambda x: model_order.get(str(x).lower(), 99))

        sort_cols = ['_scenario_rank', '_model_rank']
        ascending = [True, True]

        if 'ham_mel_recall' in df.columns:
            sort_cols.append('ham_mel_recall')
            ascending.append(False)
        if 'mel_recall' in df.columns:
            sort_cols.append('mel_recall')
            ascending.append(False)

        df.sort_values(by=sort_cols, ascending=ascending, inplace=True)
        df.drop(columns=['_scenario_rank', '_model_rank'], inplace=True, errors='ignore')
    else:
        return

    summary_md = session_dir / 'SUMMARY.md'
    with open(summary_md, 'w') as f:
        f.write(f"# 🏆 Skin Lesion Benchmark: Session Leaderboard ({session_dir.name})\n\n")
        f.write(f"*Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write("### 📊 Top Performing Hyperparameter Configurations (Dual-Domain)\n\n")

        cols = [
            'scenario', 'model',
            'ham_accuracy', 'accuracy', 'domain_gap',
            'ham_mel_auc_roc', 'mel_auc_roc',
            'ham_mel_recall', 'mel_recall',
            'ham_weighted_f1', 'weighted_avg_f1',
            'epochs', 'lr_stage2', 'patience', 'seed'
        ]
        available_cols = [c for c in cols if c in df.columns]
        display_df = df[available_cols].copy()
        f.write(display_df.to_markdown(index=False))
        f.write("\n\n---\n")


def update_global_archive_index(root_dir: Path):
    """Generates a global master catalog of all dated experiment sessions."""
    archive_file = root_dir / 'GLOBAL_ARCHIVE_INDEX.md'
    session_dirs = [d for d in root_dir.iterdir() if d.is_dir() and d.name != 'scenarios']
    session_dirs.sort(key=lambda x: x.name, reverse=True)

    with open(archive_file, 'w') as f:
        f.write("# 🗂️ Global Experiments Archive Index\n\n")
        f.write(f"*Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write("| Session Date / Folder | Runs Completed | Top In-Domain Acc (HAM) | Top Out-of-Domain Acc (PAD) | Link |\n")
        f.write("|:---|:---:|:---:|:---:|:---|\n")

        for s_dir in session_dirs:
            lb_file = s_dir / 'master_leaderboard.csv'
            if lb_file.exists():
                try:
                    df = pd.read_csv(lb_file)
                    n_runs = len(df)
                    top_ham = f"{df['ham_accuracy'].max():.2%}" if 'ham_accuracy' in df.columns and not df['ham_accuracy'].isna().all() else "N/A"
                    pad_col = 'pad_accuracy' if 'pad_accuracy' in df.columns else ('accuracy' if 'accuracy' in df.columns else None)
                    top_pad = f"{df[pad_col].max():.2%}" if pad_col and not df[pad_col].isna().all() else "N/A"
                except Exception:
                    n_runs, top_ham, top_pad = 0, "N/A", "N/A"
            else:
                n_runs, top_ham, top_pad = 0, "N/A", "N/A"

            f.write(f"| **`{s_dir.name}`** | {n_runs} | {top_ham} | {top_pad} | [`SUMMARY.md`]({s_dir.name}/SUMMARY.md) |\n")


class TeeLogger:
    """Mirrors console output simultaneously to terminal and session log file."""
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'a', encoding='utf-8', buffering=1)

    def write(self, message):
        try:
            self.terminal.write(message)
            self.terminal.flush()
        except Exception:
            pass
        try:
            self.log.write(message)
            self.log.flush()
        except Exception:
            pass

    def flush(self):
        try:
            self.terminal.flush()
        except Exception:
            pass
        try:
            self.log.flush()
        except Exception:
            pass


def main():
    args = parse_args()
    root_dir = Path(args.experiments_root)
    root_dir.mkdir(parents=True, exist_ok=True)

    session_name, session_dir = resolve_session_directory(root_dir, args.session_id, args.resume)
    session_dir.mkdir(parents=True, exist_ok=True)

    # Automatically log all console output into session folder
    session_log_file = session_dir / 'execution.log'
    tee = TeeLogger(session_log_file)
    sys.stdout = tee
    sys.stderr = tee

    with open(args.scenarios_file) as f:
        scenarios_data = json.load(f)['scenarios']

    if args.scenario == 'all':
        scenario_keys = [k for k, v in sorted(scenarios_data.items(), key=lambda x: x[1].get('priority', 99)) if not v.get('optional', False)]
    else:
        scenario_keys = [args.scenario]

    # Global cached datasets
    data_cache = Path('./data_cache')
    prepared_dir = Path('./dataset_treino')
    train_csv = root_dir / 'train_df.csv'
    ham_val_csv = root_dir / 'ham_val_df.csv'
    pad_val_csv = root_dir / 'pad_val_df.csv'

    if not (train_csv.exists() and ham_val_csv.exists() and pad_val_csv.exists()):
        print("\n  [Setup] Initializing stratified training & dual validation datasets...")
        from dataset import prepare_dataset, ensure_pad_ufes20_download, load_pad_ufes20_validation
        t_df, h_df = prepare_dataset(cache_root=data_cache, prepared_dir=prepared_dir, random_state=42)
        pad_dir = ensure_pad_ufes20_download(data_cache)
        p_df = load_pad_ufes20_validation(pad_dir)

        t_df.to_csv(train_csv, index=False)
        h_df.to_csv(ham_val_csv, index=False)
        p_df.to_csv(pad_val_csv, index=False)

    # Copy split info into session folder for permanent record
    if not (session_dir / 'train_df.csv').exists():
        t_df = pd.read_csv(train_csv)
        t_df.to_csv(session_dir / 'train_df.csv', index=False)
    if not (session_dir / 'ham_val_df.csv').exists():
        h_df = pd.read_csv(ham_val_csv)
        h_df.to_csv(session_dir / 'ham_val_df.csv', index=False)
    if not (session_dir / 'pad_val_df.csv').exists():
        p_df = pd.read_csv(pad_val_csv)
        p_df.to_csv(session_dir / 'pad_val_df.csv', index=False)

    print(f"\n{'='*85}")
    print(f" 🔬 Thesis Scenario Runner Initialized (Session: {session_name})")
    print(f" Execution Order: {' -> '.join([s.upper() for s in scenario_keys])}")
    print(f" Output Location: {session_dir.resolve()}")
    print(f" Log File:        {session_log_file.resolve()}")
    print(f"{'='*85}\n")

    for s_key in scenario_keys:
        s_info = scenarios_data[s_key]
        print(f"\n{'#'*85}")
        print(f" 🎯 STARTING SCENARIO: {s_info['name'].upper()}")
        print(f" Description: {s_info['description']}")
        print(f"{'#'*85}\n")

        models_dict = s_info['models']
        target_models = args.models if args.models else list(models_dict.keys())

        for m_idx, model_name in enumerate(target_models, start=1):
            if model_name not in models_dict:
                continue

            cfg = models_dict[model_name]
            epochs = cfg['epochs']
            bs = cfg['batch_size']
            lr1 = cfg['lr_stage1']
            lr2 = cfg['lr_stage2']
            pat = cfg['patience']
            seed = cfg['seed']

            exp_id = f"{s_key}_{model_name}_ep{epochs}_bs{bs}_lr2_{lr2}_pat{pat}_seed{seed}"
            exp_dir = session_dir / 'scenarios' / s_key / model_name
            exp_dir.mkdir(parents=True, exist_ok=True)

            print(f"[{m_idx}/{len(target_models)}] Scenario: {s_key.upper()} | Model: {model_name.upper()}")

            # ── 1. Fault Tolerance: Check if Completed ──
            if is_experiment_completed(exp_dir):
                print(f"  ⏭️ [ALREADY COMPLETED] Skipping {exp_id}. Results intact at: {exp_dir}\n")
                continue

            # ── 2. Save Exact Configuration ──
            config_data = {
                'experiment_id': exp_id,
                'session_id': session_name,
                'scenario': s_key,
                'model': model_name,
                'epochs': epochs,
                'batch_size': bs,
                'lr_stage1': lr1,
                'lr_stage2': lr2,
                'patience': pat,
                'seed': seed,
                'val_dataset': 'dual (HAM10000 + PAD-UFES-20)',
                'timestamp_start': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'python_version': sys.version.split()[0],
                'device_info': 'Auto-detected by Accelerator Engine'
            }
            with open(exp_dir / 'config.json', 'w') as f:
                json.dump(config_data, f, indent=2)

            print(f"  ▶️ [TRAINING] Epochs={epochs}, LR2={lr2}, BatchSize={bs}, Patience={pat}")

            cmd = [
                args.python_bin, 'train_timm_models.py',
                '--model', model_name,
                '--epochs', str(epochs),
                '--batch-size', str(bs),
                '--lr-stage1', str(lr1),
                '--lr-stage2', str(lr2),
                '--patience', str(pat),
                '--train-csv', str(train_csv),
                '--val-csv', str(ham_val_csv),
                '--external-val-csv', str(pad_val_csv),
                '--val-dataset', 'both',
                '--output-dir', str(exp_dir),
                '--seed', str(seed),
            ]

            t0 = time.perf_counter()
            try:
                env = os.environ.copy()
                env['PYTHONUNBUFFERED'] = '1'
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env
                )
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                process.wait()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, cmd)

                duration = time.perf_counter() - t0

                # Reorganize model output files cleanly
                sub_model_dir = exp_dir / model_name
                if sub_model_dir.exists():
                    for p in sub_model_dir.glob('*'):
                        p.rename(exp_dir / p.name)
                    try:
                        sub_model_dir.rmdir()
                    except Exception:
                        pass

                results_path = exp_dir / 'results.json'

                if results_path.exists():
                    with open(results_path) as f:
                        res_data = json.load(f)

                    ham_acc = res_data.get('ham_accuracy', 0.0)
                    pad_acc = res_data.get('accuracy', 0.0)
                    ham_w_f1 = res_data.get('ham_weighted_f1', 0.0)
                    pad_w_f1 = res_data.get('weighted_avg_f1', 0.0)
                    ham_mel_auc = res_data.get('ham_mel_auc_roc', 0.0)
                    pad_mel_auc = res_data.get('pad_mel_auc_roc', res_data.get('mel_auc_roc', 0.0))
                    gap = res_data.get('domain_gap_drop', 0.0)

                    print(f"  ✅ [SUCCESS] In-Domain: Acc={ham_acc:.2%}, Mel AUC={ham_mel_auc:.4f} | Out-of-Domain: Acc={pad_acc:.2%}, Mel AUC={pad_mel_auc:.4f} | Domain Gap: -{gap*100:.2f}% | Duration: {duration/60:.1f} min")

                    entry = {
                        'experiment_id': exp_id,
                        'session_id': session_name,
                        'scenario': s_key,
                        'model': model_name,
                        'epochs': epochs,
                        'batch_size': bs,
                        'lr_stage1': lr1,
                        'lr_stage2': lr2,
                        'patience': pat,
                        'seed': seed,
                        'val_dataset': 'dual (HAM10000 + PAD-UFES-20)',

                        # In-Domain (HAM10000)
                        'ham_accuracy': round(ham_acc, 4),
                        'ham_weighted_f1': round(ham_w_f1, 4),
                        'ham_macro_f1': round(res_data.get('ham_macro_f1', 0.0), 4),
                        'ham_mel_recall': round(res_data.get('ham_mel_recall', 0.0), 4),
                        'ham_mel_auc_roc': round(ham_mel_auc, 4),
                        'ham_macro_auc_roc': round(res_data.get('ham_macro_auc_roc', 0.0), 4),
                        'ham_harmonized_5class_acc': round(res_data.get('ham_harmonized_5class_acc', ham_acc), 4),
                        'ham_bcc_recall': round(res_data.get('ham_bcc_recall', 0.0), 4),

                        # Out-of-Domain (PAD-UFES-20)
                        'accuracy': round(pad_acc, 4),
                        'pad_accuracy': round(pad_acc, 4),
                        'weighted_avg_f1': round(pad_w_f1, 4),
                        'macro_avg_f1': round(res_data.get('macro_avg_f1', 0.0), 4),
                        'mel_recall': round(res_data.get('mel_recall', 0.0), 4),
                        'mel_auc_roc': round(pad_mel_auc, 4),
                        'pad_mel_auc_roc': round(pad_mel_auc, 4),
                        'macro_auc_roc': round(res_data.get('macro_auc_roc', 0.0), 4),
                        'pad_macro_auc_roc': round(res_data.get('pad_macro_auc_roc', 0.0), 4),
                        'harmonized_5class_acc': round(res_data.get('pad_harmonized_5class_acc', res_data.get('harmonized_5class_acc', pad_acc)), 4),
                        'bcc_recall': round(res_data.get('bcc_recall', 0.0), 4),

                        # Domain Gap
                        'domain_gap': round(gap, 4),

                        # Stage 1 Benchmarks
                        'stage1_ham_acc': round(res_data.get('stage1_ham_accuracy', 0.0), 4),
                        'stage1_pad_acc': round(res_data.get('stage1_pad_accuracy', 0.0), 4),

                        'duration_min': round(duration / 60, 2),
                        'completed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'status': 'SUCCESS'
                    }

                    # Update per-scenario tracker & session leaderboard
                    scenario_csv = session_dir / 'scenarios' / s_key / 'history_runs.csv'
                    if scenario_csv.exists():
                        s_df = pd.read_csv(scenario_csv)
                        s_df = s_df[s_df['experiment_id'] != exp_id]
                        s_df = pd.concat([s_df, pd.DataFrame([entry])], ignore_index=True)
                    else:
                        s_df = pd.DataFrame([entry])
                    s_df.sort_values(by='ham_accuracy', ascending=False).to_csv(scenario_csv, index=False)

                    update_master_leaderboard(session_dir, entry)
                    update_global_archive_index(root_dir)
                    print(f"  📁 Session Leaderboard updated: {session_dir / 'master_leaderboard.csv'}\n")

            except subprocess.CalledProcessError as e:
                print(f"  ❌ [FAILED] Error running {model_name}: {e}\n")

    update_master_leaderboard(session_dir)
    update_global_archive_index(root_dir)
    print(f"\n{'='*85}")
    print(f" 🎉 All Scenarios Finished in Session '{session_name}'!")
    print(f" 📊 Session Leaderboard: {session_dir / 'master_leaderboard.csv'}")
    print(f" 📄 Session Report:      {session_dir / 'SUMMARY.md'}")
    print(f" 🗂️ Global Archive:      {root_dir / 'GLOBAL_ARCHIVE_INDEX.md'}")
    print(f"{'='*85}\n")


if __name__ == '__main__':
    main()
