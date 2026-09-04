"""
Automated Grid Search & Experiment Tracking Suite for Skin Lesion Benchmarking.

Organizes experiments neatly per model:
  experiments/
  ├── master_leaderboard.csv            # Cross-model ranking sorted by accuracy & F1
  ├── SUMMARY.md                        # Auto-generated markdown report for your thesis
  ├── v1/
  │   ├── history_runs.csv              # Track every combination tested for MobileNet V1
  │   ├── best_configuration.json       # Best hyperparameter set found for V1
  │   └── exp_001_ep30_bs32_lr1e-4.../  # Artifacts (best_model.pth, plots, reports)
  ├── v2/
  ├── v3large/
  ├── v4conv/
  └── v5/
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Structured Experiment Orchestrator & Tracker")
    parser.add_argument('--allow-grid-search', action='store_true', default=False,
                        help="Explicitly enable extra hyperparameter runs outside the final thesis protocol")
    parser.add_argument('--models', nargs='+', default=['v1', 'v2', 'v3large', 'v4conv', 'v5'],
                        help="Models to benchmark (e.g. v1 v2 v3large v4conv v5)")
    parser.add_argument('--epochs-list', nargs='+', type=int, default=[15, 30],
                        help="List of epoch counts to test (e.g. 15 30)")
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[32],
                        help="List of batch sizes to test (e.g. 16 32)")
    parser.add_argument('--lr-stage1-list', nargs='+', type=float, default=[1e-3],
                        help="List of Stage 1 learning rates")
    parser.add_argument('--lr-stage2-list', nargs='+', type=float, default=[1e-4, 5e-5],
                        help="List of Stage 2 learning rates")
    parser.add_argument('--patience-list', nargs='+', type=int, default=[5, 8],
                        help="List of EarlyStopping patience values")
    parser.add_argument('--seeds', nargs='+', type=int, default=[42],
                        help="List of random seeds (e.g. 42 123)")
    parser.add_argument('--val-dataset', type=str, default='ham10000',
                        choices=['ham10000', 'pad-ufes-20'], help="Hyperparameter selection dataset")
    parser.add_argument('--allow-pad-selection', action='store_true', default=False,
                        help="Explicitly permit methodologically invalid PAD-based selection")
    parser.add_argument('--experiments-root', type=str, default='./experiments',
                        help="Root directory where all organized data will be saved")
    parser.add_argument('--python-bin', type=str, default=sys.executable,
                        help="Path to Python executable in virtual environment")
    return parser.parse_args()


def generate_exp_dirname(epochs, batch_size, lr1, lr2, patience, seed):
    """Generate a clean directory name for an individual experiment."""
    return f"ep{epochs}_bs{batch_size}_lr1_{lr1}_lr2_{lr2}_pat{patience}_seed{seed}"


def is_experiment_completed(exp_dir: Path) -> bool:
    """Check if an experiment has already finished and produced valid results."""
    res_file = exp_dir / 'results.json'
    cfg_file = exp_dir / 'config.json'
    return res_file.exists() and cfg_file.exists() and res_file.stat().st_size > 10


def update_model_tracker(model_dir: Path, new_entry: dict):
    """Update the per-model history CSV and best_configuration.json."""
    history_csv = model_dir / 'history_runs.csv'
    if history_csv.exists():
        df = pd.read_csv(history_csv)
        df = df[df['experiment_name'] != new_entry['experiment_name']]
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([new_entry])

    df.sort_values(by='selection_mel_auc_roc', ascending=False, inplace=True)
    df.to_csv(history_csv, index=False)

    # Save best configuration for this model
    best_row = df.iloc[0].to_dict()
    with open(model_dir / 'best_configuration.json', 'w') as f:
        json.dump(best_row, f, indent=2)


def update_master_leaderboard(root_dir: Path, new_entry: dict = None):
    """Update cross-model master leaderboard CSV and markdown summary."""
    leaderboard_csv = root_dir / 'master_leaderboard.csv'
    if new_entry is not None:
        if leaderboard_csv.exists():
            df = pd.read_csv(leaderboard_csv)
            df = df[df['experiment_id'] != new_entry['experiment_id']]
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        else:
            df = pd.DataFrame([new_entry])

        df.sort_values(by=['selection_mel_auc_roc', 'pad_mel_auc_roc'], ascending=[False, False], inplace=True)
        df.to_csv(leaderboard_csv, index=False)
    elif leaderboard_csv.exists():
        df = pd.read_csv(leaderboard_csv)
    else:
        return

    # Generate Markdown Summary for Thesis
    summary_md = root_dir / 'SUMMARY.md'
    with open(summary_md, 'w') as f:
        f.write("# 🏆 Skin Lesion Benchmark: Master Experiments Leaderboard\n\n")
        f.write(f"*Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write("### 📊 Top Performing Hyperparameter Configurations\n\n")

        cols = ['model', 'accuracy', 'weighted_avg_f1', 'macro_avg_f1', 'mel_recall', 'bcc_recall', 'epochs', 'lr_stage2', 'patience', 'seed']
        available_cols = [c for c in cols if c in df.columns]
        display_df = df[available_cols].copy()
        f.write(display_df.to_markdown(index=False))
        f.write("\n\n---\n")


def main():
    args = parse_args()
    if not args.allow_grid_search:
        raise SystemExit("Grid search is disabled for the fixed-budget thesis protocol; pass --allow-grid-search explicitly")
    if args.val_dataset == 'pad-ufes-20' and not args.allow_pad_selection:
        raise SystemExit("PAD-UFES-20 cannot be used for hyperparameter selection without --allow-pad-selection")
    if args.seeds != [42]:
        raise SystemExit("This thesis pipeline uses the single fixed seed 42")
    root_dir = Path(args.experiments_root)
    root_dir.mkdir(parents=True, exist_ok=True)

    combinations = list(itertools.product(
        args.models,
        args.epochs_list,
        args.batch_sizes,
        args.lr_stage1_list,
        args.lr_stage2_list,
        args.patience_list,
        args.seeds
    ))

    total_runs = len(combinations)
    print(f"\n{'='*80}")
    print(f" 📂 Structured Experiment Tracking & Grid Search")
    print(f" Root Experiments Directory: {root_dir.resolve()}")
    print(f" Total Combinations in Queue: {total_runs}")
    print(f" Validation Dataset: {args.val_dataset.upper()}")
    print(f"{'='*80}\n")

    # Ensure dataset splits exist
    data_cache = Path('./data_cache')
    prepared_dir = Path('./dataset_treino')
    train_csv = root_dir / 'train_df.csv'
    val_csv = root_dir / 'ham_val_df.csv'
    test_csv = root_dir / 'ham_test_df.csv'
    pad_csv = root_dir / 'pad_val_df.csv'

    if not all(path.exists() for path in (train_csv, val_csv, test_csv, pad_csv)):
        print("  [Setup] Initializing train, HAM validation/test, and PAD evaluation datasets...")
        from dataset import ensure_pad_ufes20_download, load_pad_ufes20_validation, prepare_dataset
        t_df, v_df, test_df = prepare_dataset(
            cache_root=data_cache, prepared_dir=prepared_dir, random_state=42, oversample=False
        )
        pad_df = load_pad_ufes20_validation(ensure_pad_ufes20_download(data_cache))
        t_df.to_csv(train_csv, index=False)
        v_df.to_csv(val_csv, index=False)
        test_df.to_csv(test_csv, index=False)
        pad_df.to_csv(pad_csv, index=False)
        print(f"  [Setup] Datasets cached under {root_dir}\n")

    for run_idx, (model, epochs, bs, lr1, lr2, pat, seed) in enumerate(combinations, start=1):
        model_dir = root_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)

        exp_name = generate_exp_dirname(epochs, bs, lr1, lr2, pat, seed)
        exp_id = f"{model}_{exp_name}"
        exp_dir = model_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{run_idx}/{total_runs}] Model: {model.upper()} | Config: {exp_name}")

        # ── 1. Auto-Resume / Fault Tolerance ──
        if is_experiment_completed(exp_dir):
            print(f"  ⏭️ [ALREADY RECORDED] Skipping execution. Results intact at: {exp_dir}\n")
            continue

        # ── 2. Record Input Configuration ──
        config_data = {
            'experiment_id': exp_id,
            'experiment_name': exp_name,
            'model': model,
            'epochs': epochs,
            'batch_size': bs,
            'lr_stage1': lr1,
            'lr_stage2': lr2,
            'patience': pat,
            'seed': seed,
            'val_dataset': args.val_dataset,
            'timestamp_start': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'python_version': sys.version.split()[0],
            'device_info': 'Auto-detected by Accelerator Engine'
        }
        with open(exp_dir / 'config.json', 'w') as f:
            json.dump(config_data, f, indent=2)

        print(f"  ▶️ [STARTING TRAINING] (Outputs saved to: {exp_dir})")

        cmd = [
            args.python_bin, 'train_timm_models.py',
            '--model', model,
            '--epochs', str(epochs),
            '--batch-size', str(bs),
            '--lr-stage1', str(lr1),
            '--lr-stage2', str(lr2),
            '--patience', str(pat),
            '--train-csv', str(train_csv),
            '--val-csv', str(val_csv),
            '--test-csv', str(test_csv),
            '--external-val-csv', str(pad_csv),
            '--output-dir', str(exp_dir),
            '--seed', str(seed),
        ]

        t0 = time.perf_counter()
        try:
            subprocess.run(cmd, check=True)
            duration = time.perf_counter() - t0

            # Parse results from model subfolder
            results_path = exp_dir / model / 'results.json'

            if results_path.exists():
                with open(results_path) as f:
                    res_data = json.load(f)

                # Move files directly to exp_dir for clean access
                for p in (exp_dir / model).glob('*'):
                    p.rename(exp_dir / p.name)
                try:
                    (exp_dir / model).rmdir()
                except Exception:
                    pass

                acc = res_data.get('pad_accuracy', 0.0)
                w_f1 = res_data.get('pad_weighted_f1', 0.0)
                m_f1 = res_data.get('pad_macro_f1', 0.0)
                mel_rec = res_data.get('pad_mel_recall', 0.0)
                bcc_rec = res_data.get('pad_bcc_recall', 0.0)
                akiec_rec = res_data.get('pad_akiec_recall', 0.0)
                selection_auc = res_data.get('ham_val_mel_auc_roc', 0.0)
                pad_auc = res_data.get('pad_mel_auc_roc', 0.0)

                print(f"  ✅ [COMPLETED in {duration/60:.1f} min] Acc: {acc:.2%}, Weighted F1: {w_f1:.4f}, Mel Recall: {mel_rec:.2%}")

                entry = {
                    'experiment_id': exp_id,
                    'experiment_name': exp_name,
                    'model': model,
                    'epochs': epochs,
                    'batch_size': bs,
                    'lr_stage1': lr1,
                    'lr_stage2': lr2,
                    'patience': pat,
                    'seed': seed,
                    'val_dataset': args.val_dataset,
                    'selection_mel_auc_roc': round(selection_auc, 4),
                    'pad_mel_auc_roc': round(pad_auc, 4),
                    'accuracy': round(acc, 4),
                    'weighted_avg_f1': round(w_f1, 4),
                    'macro_avg_f1': round(m_f1, 4),
                    'mel_recall': round(mel_rec, 4),
                    'bcc_recall': round(bcc_rec, 4),
                    'akiec_recall': round(akiec_rec, 4),
                    'duration_min': round(duration / 60, 2),
                    'completed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'status': 'SUCCESS'
                }

                # Update trackers
                update_model_tracker(model_dir, entry)
                update_master_leaderboard(root_dir, entry)
                print(f"  📁 Trackers updated: {model_dir / 'history_runs.csv'} & {root_dir / 'master_leaderboard.csv'}\n")

        except subprocess.CalledProcessError as e:
            print(f"  ❌ [FAILED] Error during execution: {e}\n")

    update_master_leaderboard(root_dir)
    print(f"\n{'='*80}")
    print(f" 🎉 All Scheduled Experiments Finished Successfully!")
    print(f" 📊 Master Leaderboard: {root_dir / 'master_leaderboard.csv'}")
    print(f" 📄 Summary Report:     {root_dir / 'SUMMARY.md'}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
