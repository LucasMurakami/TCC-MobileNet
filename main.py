"""
Unified CLI Entrypoint & Multi-Model Benchmark Runner.
Handles argument parsing, dynamic benchmark_scenarios.json configuration,
automatic session/scenario directory management, dataset preparation,
logging/prints, benchmark summarization, and archive indexing.
Delegates core training and modeling routines to the timm backbone engine (train_timm_models.py).
"""

import argparse
from datetime import datetime
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from dataset import (
    prepare_dataset, ensure_pad_ufes20_download, load_pad_ufes20_validation
)
from train_timm_models import (
    MODEL_CONFIGS, configure_hardware_environment, train_single_model
)
from visualize import plot_benchmark_summary


def load_benchmark_scenarios(scenarios_file: Union[str, Path] = "benchmark_scenarios.json") -> dict:
    """
    Loads benchmark scenarios and per-model hyperparameter presets dynamically from JSON.
    Supports both 'benchmark_scenarios.json' and 'benchmark_scenarions.json'.
    """
    target_path = Path(scenarios_file)
    if not target_path.exists():
        alt_path = Path("benchmark_scenarions.json") if target_path.name == "benchmark_scenarios.json" else Path("benchmark_scenarios.json")
        if alt_path.exists():
            target_path = alt_path

    if target_path.exists():
        try:
            with open(target_path, 'r') as f:
                data = json.load(f)
                return data.get('scenarios', data)
        except Exception as e:
            print(f"⚠️ Warning: Failed reading scenario config from {target_path}: {e}")

    # Fallback standard configurations
    return {
        "standard": {
            "name": "Standard (20 Epochs Production)",
            "models": {
                "v1": {"epochs": 20, "batch_size": 32, "lr_stage1": 0.001, "lr_stage2": 0.0001, "patience": 4, "seed": 42},
                "v2": {"epochs": 20, "batch_size": 32, "lr_stage1": 0.001, "lr_stage2": 0.0001, "patience": 4, "seed": 42},
                "v3": {"epochs": 20, "batch_size": 32, "lr_stage1": 0.001, "lr_stage2": 0.0001, "patience": 4, "seed": 42},
                "v4": {"epochs": 20, "batch_size": 32, "lr_stage1": 0.001, "lr_stage2": 0.00005, "patience": 4, "seed": 42},
                "v5": {"epochs": 20, "batch_size": 32, "lr_stage1": 0.0005, "lr_stage2": 0.00002, "patience": 4, "seed": 42}
            }
        },
        "medium": {
            "name": "Medium (Balanced Research Budget)",
            "models": {
                "v1": {"epochs": 30, "batch_size": 32, "lr_stage1": 0.001, "lr_stage2": 0.0001, "patience": 5, "seed": 42},
                "v2": {"epochs": 30, "batch_size": 32, "lr_stage1": 0.001, "lr_stage2": 0.0001, "patience": 5, "seed": 42},
                "v3": {"epochs": 30, "batch_size": 32, "lr_stage1": 0.001, "lr_stage2": 0.0001, "patience": 5, "seed": 42},
                "v4": {"epochs": 30, "batch_size": 32, "lr_stage1": 0.001, "lr_stage2": 0.00005, "patience": 5, "seed": 42},
                "v5": {"epochs": 30, "batch_size": 32, "lr_stage1": 0.0005, "lr_stage2": 0.00002, "patience": 5, "seed": 42}
            }
        },
        "low": {
            "name": "Low (Fast Exploration)",
            "models": {
                "v1": {"epochs": 15, "batch_size": 32, "lr_stage1": 0.001, "lr_stage2": 0.0001, "patience": 3, "seed": 42},
                "v2": {"epochs": 15, "batch_size": 32, "lr_stage1": 0.001, "lr_stage2": 0.0001, "patience": 3, "seed": 42},
                "v3": {"epochs": 15, "batch_size": 32, "lr_stage1": 0.001, "lr_stage2": 0.0001, "patience": 3, "seed": 42},
                "v4": {"epochs": 15, "batch_size": 32, "lr_stage1": 0.001, "lr_stage2": 0.00005, "patience": 3, "seed": 42},
                "v5": {"epochs": 15, "batch_size": 32, "lr_stage1": 0.0005, "lr_stage2": 0.00002, "patience": 3, "seed": 42}
            }
        },
        "maximum": {
            "name": "Maximum (Peak Performance)",
            "models": {
                "v1": {"epochs": 50, "batch_size": 32, "lr_stage1": 0.001, "lr_stage2": 0.0001, "patience": 8, "seed": 42},
                "v2": {"epochs": 50, "batch_size": 32, "lr_stage1": 0.001, "lr_stage2": 0.0001, "patience": 8, "seed": 42},
                "v3": {"epochs": 50, "batch_size": 32, "lr_stage1": 0.001, "lr_stage2": 0.0001, "patience": 8, "seed": 42},
                "v4": {"epochs": 50, "batch_size": 32, "lr_stage1": 0.001, "lr_stage2": 0.00005, "patience": 8, "seed": 42},
                "v5": {"epochs": 50, "batch_size": 32, "lr_stage1": 0.0005, "lr_stage2": 0.00002, "patience": 8, "seed": 42}
            }
        }
    }


def print_banner(hw: dict, session_dir: Path, scenario: str):
    """Prints a styled environment, session & hardware status banner."""
    print("=" * 85)
    print(" 🔬 Skin Lesion Classification: MobileNet (V1–V5) Benchmark Suite")
    print(f" 📂 Session Dir:  {session_dir}")
    print(f" 🎯 Scenario:     {scenario.upper()}")
    print(f" ⚙️  Hardware:     {hw['device_name']} ({hw['vram_gb']:.1f} GB VRAM, {hw['device_count']} GPU(s))")
    print(f" 🚀 Precision:    {hw['precision_name']}")
    print("=" * 85)


def print_leaderboard(all_results: dict):
    """Prints a formatted benchmark leaderboard for multi-model runs."""
    print("\n" + "=" * 85)
    print(f"  {'Model':<10} {'HAM10000 Acc':<15} {'PAD-UFES Acc':<15} {'Mel Recall (PAD)':<18} {'Mel AUC-ROC':<15}")
    print("=" * 85)
    for m in all_results:
        ham_acc = all_results[m].get('ham_accuracy', 0.0)
        pad_acc = all_results[m].get('pad_accuracy', 0.0)
        mel_rec = all_results[m].get('pad_mel_recall', 0.0)
        mel_auc = all_results[m].get('pad_mel_auc_roc', 0.0)
        print(f"  {m:<10} {ham_acc:<15.2%} {pad_acc:<15.2%} {mel_rec:<18.2%} {mel_auc:<15.4f}")
    print("=" * 85)
    best_m = max(all_results, key=lambda k: all_results[k].get('pad_accuracy', 0.0))
    print(f"🏆 Best Overall Out-of-Domain Model: {best_m.upper()} (PAD-UFES-20 Acc: {all_results[best_m]['pad_accuracy']:.2%})\n")


def update_session_leaderboard(session_dir: Path) -> pd.DataFrame:
    """
    Scans all results.json files in a session directory, generates/updates
    master_leaderboard.csv, SUMMARY.md, and benchmark comparison artifacts.
    """
    session_dir = Path(session_dir)
    results_files = list(session_dir.rglob('results.json'))
    if not results_files:
        return pd.DataFrame()

    records = []
    all_results_dict = {}

    for rf in results_files:
        try:
            with open(rf, 'r') as f:
                data = json.load(f)

            # Determine scenario and model name from path
            rel_parts = rf.relative_to(session_dir).parts
            if len(rel_parts) >= 3 and rel_parts[0] == 'scenarios':
                scenario_name = rel_parts[1]
                model_name = rel_parts[2]
            elif len(rel_parts) >= 2:
                scenario_name = 'standard'
                model_name = rel_parts[0]
            else:
                scenario_name = 'standard'
                model_name = data.get('model', 'unknown')

            ham_acc = float(data.get('ham_accuracy', 0.0))
            pad_acc = float(data.get('pad_accuracy', 0.0))
            gap = float(data.get('domain_gap', data.get('domain_gap_drop', ham_acc - pad_acc)))

            records.append({
                'scenario': scenario_name,
                'model': model_name,
                'ham_accuracy': round(ham_acc, 4),
                'accuracy': round(pad_acc, 4),
                'domain_gap': round(gap, 4),
                'ham_mel_auc_roc': round(float(data.get('ham_mel_auc_roc', 0.0)), 4),
                'mel_auc_roc': round(float(data.get('pad_mel_auc_roc', 0.0)), 4),
                'triage_th': round(float(data.get('mel_triage_threshold', 0.15)), 2) if data.get('mel_triage_threshold') is not None else 0.15,
                'ham_triage_sens': round(float(data.get('ham_mel_triage_recall', data.get('ham_mel_recall', 0.0))), 4),
                'pad_triage_sens': round(float(data.get('pad_mel_triage_recall', data.get('pad_mel_recall', 0.0))), 4),
                'ham_mel_recall': round(float(data.get('ham_mel_recall', 0.0)), 4),
                'mel_recall': round(float(data.get('pad_mel_recall', 0.0)), 4),
                'bcc_th': round(float(data.get('bcc_triage_threshold', 0.15)), 2) if data.get('bcc_triage_threshold') is not None else 0.15,
                'ham_bcc_triage_sens': round(float(data.get('ham_bcc_triage_recall', data.get('ham_bcc_recall', 0.0))), 4),
                'pad_bcc_triage_sens': round(float(data.get('pad_bcc_triage_recall', data.get('pad_bcc_recall', 0.0))), 4),
                'ham_bcc_recall': round(float(data.get('ham_bcc_recall', 0.0)), 4),
                'bcc_recall': round(float(data.get('pad_bcc_recall', 0.0)), 4),
                'pad_mal_screen_sens': round(float(data.get('pad_malignant_triage_recall', 0.0)), 4) if data.get('pad_malignant_triage_recall') is not None else None,
                'ham_weighted_f1': round(float(data.get('ham_weighted_f1', 0.0)), 4),
                'weighted_avg_f1': round(float(data.get('pad_weighted_f1', 0.0)), 4),
            })
            all_results_dict[f"{scenario_name}_{model_name}" if len(results_files) > 5 else model_name] = data
        except Exception as e:
            print(f"Warning: could not parse {rf}: {e}")

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(by='accuracy', ascending=False)
        df.to_csv(session_dir / 'master_leaderboard.csv', index=False)

        # Markdown Leaderboard Table
        header = '| ' + ' | '.join(df.columns) + ' |\n| ' + ' | '.join([':---' for _ in df.columns]) + ' |\n'
        rows = '\n'.join(['| ' + ' | '.join(str(val) for val in row) + ' |' for row in df.values])
        table_md = header + rows

        summary_md = f"""# 🏆 Skin Lesion Benchmark: Session Leaderboard ({session_dir.name})

*Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

### 📊 Top Performing Hyperparameter Configurations (Dual-Domain)

{table_md}

---
"""
        with open(session_dir / 'SUMMARY.md', 'w') as f:
            f.write(summary_md)

        # Update summary_comparison.json & plot if applicable
        if len(all_results_dict) > 1:
            with open(session_dir / 'summary_comparison.json', 'w') as f:
                json.dump(all_results_dict, f, indent=2)
            try:
                plot_benchmark_summary(all_results_dict, session_dir / 'benchmark_comparison.png')
            except Exception:
                pass

    return df


def update_global_archive_index(experiments_dir: Path = Path('experiments')):
    """
    Scans all session directories in experiments/, compiles performance summaries,
    and automatically updates experiments/GLOBAL_ARCHIVE_INDEX.md.
    """
    experiments_dir = Path(experiments_dir)
    if not experiments_dir.exists():
        return

    sessions = []
    for p in sorted(experiments_dir.iterdir(), reverse=True):
        if p.is_dir() and re.match(r'^\d{2}_\d{2}_\d{4}', p.name):
            results = list(p.rglob('results.json'))
            run_count = len(results)
            top_ham = 0.0
            top_pad = 0.0

            for rf in results:
                try:
                    with open(rf, 'r') as f:
                        d = json.load(f)
                    h_acc = float(d.get('ham_accuracy', 0.0))
                    p_acc = float(d.get('pad_accuracy', 0.0))
                    if h_acc > top_ham:
                        top_ham = h_acc
                    if p_acc > top_pad:
                        top_pad = p_acc
                except Exception:
                    pass

            ham_str = f"{top_ham:.2%}" if top_ham > 0 else "N/A"
            pad_str = f"{top_pad:.2%}" if top_pad > 0 else "N/A"
            sessions.append({
                'name': p.name,
                'runs': run_count,
                'top_ham': ham_str,
                'top_pad': pad_str,
                'link': f"[`SUMMARY.md`]({p.name}/SUMMARY.md)"
            })

    if not sessions:
        return

    index_lines = [
        "# 🗂️ Global Experiments Archive Index",
        "",
        f"*Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "| Session Date / Folder | Runs Completed | Top In-Domain Acc (HAM) | Top Out-of-Domain Acc (PAD) | Link |",
        "|:---|:---:|:---:|:---:|:---|"
    ]
    for s in sessions:
        index_lines.append(f"| **`{s['name']}`** | {s['runs']} | {s['top_ham']} | {s['top_pad']} | {s['link']} |")

    index_lines.append("")
    with open(experiments_dir / 'GLOBAL_ARCHIVE_INDEX.md', 'w') as f:
        f.write('\n'.join(index_lines))
    print(f"📚 Updated {experiments_dir / 'GLOBAL_ARCHIVE_INDEX.md'} ({len(sessions)} session(s) cataloged)")


def parse_args():
    parser = argparse.ArgumentParser(description='Dual-Domain PyTorch Pretrained Models Trainer (MobileNet V1 - V5)')
    parser.add_argument('--model', type=str, default='v1',
                        choices=['v1', 'v2', 'v3', 'v3small', 'v3large', 'v4', 'v4conv', 'v4convl', 'v5', 'all'],
                        help='Model variant to train or "all" to benchmark all generations')
    parser.add_argument('--scenario', type=str, default='standard',
                        choices=['standard', 'medium', 'low', 'maximum', 'custom'],
                        help='Scenario preset from benchmark_scenarios.json (default: standard)')
    parser.add_argument('--scenarios-file', type=str, default='benchmark_scenarios.json',
                        help='Path to benchmark scenarios JSON configuration file')
    parser.add_argument('--session-dir', type=str, default=None,
                        help='Session directory (default: experiments/<DD_MM_YYYY>)')
    parser.add_argument('--epochs', type=int, default=None, help='Max epochs for Stage 2 (overrides scenario preset)')
    parser.add_argument('--patience', type=int, default=None, help='Early stopping patience (overrides scenario preset)')
    parser.add_argument('--batch-size', type=int, default=None, help='Target batch size (overrides scenario preset)')
    parser.add_argument('--lr-stage1', type=float, default=None, help='Learning rate for Stage 1 (Warmup)')
    parser.add_argument('--lr-stage2', type=float, default=None, help='Learning rate for Stage 2 (Fine-tuning)')
    parser.add_argument('--img-size', type=int, default=None, help='Input image resolution')
    parser.add_argument('--train-csv', type=str, default=None, help='Training set CSV (HAM10000 80 percent)')
    parser.add_argument('--val-csv', type=str, default=None, help='In-domain validation set CSV (HAM10000 20 percent)')
    parser.add_argument('--external-val-csv', type=str, default=None, help='Out-of-domain validation set CSV (PAD-UFES-20)')
    parser.add_argument('--cache-dir', type=str, default='./data_cache', help='Dataset cache directory')
    parser.add_argument('--prepared-dir', type=str, default='./dataset_treino', help='Prepared dataset directory')
    parser.add_argument('--pad-ufes-dir', type=str, default='./data_cache/pad_ufes_20_raw', help='PAD-UFES-20 dataset directory')
    parser.add_argument('--val-dataset', type=str, default='both', choices=['ham10000', 'pad-ufes-20', 'both'])
    parser.add_argument('--mel-threshold', type=str, default=None,
                        help="Operating sensitivity threshold for melanoma triage (e.g. 0.15, 'auto', 'youden', 'sens90', 'sens95'). CLI always overrides benchmark_scenarios.json.")
    parser.add_argument('--bcc-threshold', type=str, default=None,
                        help="Operating sensitivity threshold for Basal Cell Carcinoma (BCC) triage (e.g. 'youden', 'auto', 'sens90', 'sens95', or float like 0.15). CLI always overrides benchmark_scenarios.json.")
    parser.add_argument('--malignant-threshold', type=str, default=None,
                        help="Operating threshold for joint malignancy screening (MEL + BCC + AKIEC). Default: None.")
    parser.add_argument('--balanced-sampling', action='store_true', default=False,
                        help="Enable class-balanced mini-batch sampling via WeightedRandomSampler (eliminates 67%% Nevus gradient dominance).")
    parser.add_argument('--logit-adjust', type=float, default=None,
                        help="Post-hoc Bayesian logit adjustment strength tau (e.g. 1.0) to cancel training prior penalty on minority classes.")
    parser.add_argument('--mixup-minority', type=float, default=None,
                        help="Beta-distribution alpha parameter (e.g. 0.2) for minority class Mixup data augmentation.")
    parser.add_argument('--use-tta', action='store_true', default=False,
                        help="Enable 4-view Test-Time Augmentation (orig, hflip, vflip, rot90) during evaluation.")
    parser.add_argument('--color-constancy', action='store_true', default=False,
                        help="Enable Shades-of-Gray Minkowski color constancy transform to standardize cross-domain illumination.")
    parser.add_argument('--output-dir', type=str, default=None, help='Custom output directory (overrides auto-session)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load Scenarios dynamically from JSON
    scenarios = load_benchmark_scenarios(args.scenarios_file)

    # Resolve Session Directory and Output Directory Automatically
    today_str = datetime.now().strftime('%d_%m_%Y')
    if args.session_dir:
        session_dir = Path(args.session_dir)
    else:
        session_dir = Path('experiments') / today_str

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = session_dir / 'scenarios' / args.scenario

    output_dir.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir(parents=True, exist_ok=True)

    # Global Random Seed
    base_seed = args.seed if args.seed is not None else 42
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)

    # Hardware Detection
    hw = configure_hardware_environment()
    print_banner(hw, session_dir, args.scenario)

    # Load / Prepare Datasets
    data_cache = Path(args.cache_dir)
    prepared_dir = Path(args.prepared_dir)
    pad_ufes_dir = Path(args.pad_ufes_dir)

    # 1. Training & In-Domain Validation (HAM10000)
    if args.train_csv and Path(args.train_csv).exists() and args.val_csv and Path(args.val_csv).exists():
        train_df = pd.read_csv(args.train_csv)
        ham_val_df = pd.read_csv(args.val_csv)
    elif (session_dir / 'train_df.csv').exists() and (session_dir / 'ham_val_df.csv').exists():
        train_df = pd.read_csv(session_dir / 'train_df.csv')
        ham_val_df = pd.read_csv(session_dir / 'ham_val_df.csv')
    else:
        print("[dataset] Preparing HAM10000 dataset (zero-leakage patient-grouped split)...")
        train_df, ham_val_df = prepare_dataset(data_cache, prepared_dir, random_state=base_seed, oversample=not args.balanced_sampling)
        train_df.to_csv(session_dir / 'train_df.csv', index=False)
        ham_val_df.to_csv(session_dir / 'ham_val_df.csv', index=False)

    # 2. Out-of-Domain Validation (PAD-UFES-20)
    if args.external_val_csv and Path(args.external_val_csv).exists():
        pad_val_df = pd.read_csv(args.external_val_csv)
    elif (session_dir / 'pad_val_df.csv').exists():
        pad_val_df = pd.read_csv(session_dir / 'pad_val_df.csv')
    else:
        print("[dataset] Preparing PAD-UFES-20 validation dataset...")
        pad_dir = ensure_pad_ufes20_download(data_cache) if not pad_ufes_dir.exists() else pad_ufes_dir
        pad_val_df = load_pad_ufes20_validation(pad_dir)
        pad_val_df.to_csv(session_dir / 'pad_val_df.csv', index=False)

    # Determine Models to Train
    if args.model == 'all':
        models_to_train = ['v1', 'v2', 'v3', 'v4', 'v5']
    else:
        canonical_model = args.model
        if canonical_model in ('v4conv', 'v4convl'):
            canonical_model = 'v4'
        elif canonical_model in ('v3large', 'v3small'):
            canonical_model = 'v3'
        models_to_train = [canonical_model]

    scenario_cfg = scenarios.get(args.scenario, {})
    scenario_models_cfg = scenario_cfg.get('models', {})

    all_results = {}
    for m in models_to_train:
        # Build model-specific args from benchmark_scenarios.json with CLI overrides
        m_cfg = scenario_models_cfg.get(m, {})
        model_args = argparse.Namespace(**vars(args))

        model_args.epochs = args.epochs if args.epochs is not None else m_cfg.get('epochs', 20)
        model_args.patience = args.patience if args.patience is not None else m_cfg.get('patience', 4)
        model_args.batch_size = args.batch_size if args.batch_size is not None else m_cfg.get('batch_size', 32)
        model_args.lr_stage1 = args.lr_stage1 if args.lr_stage1 is not None else m_cfg.get('lr_stage1', None)
        model_args.lr_stage2 = args.lr_stage2 if args.lr_stage2 is not None else m_cfg.get('lr_stage2', None)
        model_args.seed = args.seed if args.seed is not None else m_cfg.get('seed', 42)

        # Melanoma Triage Threshold Resolution: CLI argument > model JSON > scenario JSON > default 0.15
        if args.mel_threshold is not None:
            model_args.mel_threshold = args.mel_threshold
        elif 'mel_threshold' in m_cfg:
            model_args.mel_threshold = m_cfg['mel_threshold']
        elif 'mel_threshold' in scenario_cfg:
            model_args.mel_threshold = scenario_cfg['mel_threshold']
        else:
            model_args.mel_threshold = 0.15

        # Basal Cell Carcinoma (BCC) Triage Threshold Resolution: CLI argument > model JSON > scenario JSON > default 'youden'
        if args.bcc_threshold is not None:
            model_args.bcc_threshold = args.bcc_threshold
        elif 'bcc_threshold' in m_cfg:
            model_args.bcc_threshold = m_cfg['bcc_threshold']
        elif 'bcc_threshold' in scenario_cfg:
            model_args.bcc_threshold = scenario_cfg['bcc_threshold']
        else:
            model_args.bcc_threshold = 'youden'

        # Malignant Screening Threshold Resolution: CLI argument > model JSON > scenario JSON > default None
        if args.malignant_threshold is not None:
            model_args.malignant_threshold = args.malignant_threshold
        elif 'malignant_threshold' in m_cfg:
            model_args.malignant_threshold = m_cfg['malignant_threshold']
        elif 'malignant_threshold' in scenario_cfg:
            model_args.malignant_threshold = scenario_cfg['malignant_threshold']
        else:
            model_args.malignant_threshold = None

        # Long-Tail Learning Resolutions (CLI flag > model JSON > scenario JSON > default)
        if args.balanced_sampling:
            model_args.balanced_sampling = True
        elif 'balanced_sampling' in m_cfg:
            model_args.balanced_sampling = m_cfg['balanced_sampling']
        elif 'balanced_sampling' in scenario_cfg:
            model_args.balanced_sampling = scenario_cfg['balanced_sampling']
        else:
            model_args.balanced_sampling = False

        if args.logit_adjust is not None:
            model_args.logit_adjust = args.logit_adjust
        elif 'logit_adjust' in m_cfg:
            model_args.logit_adjust = m_cfg['logit_adjust']
        elif 'logit_adjust' in scenario_cfg:
            model_args.logit_adjust = scenario_cfg['logit_adjust']
        else:
            model_args.logit_adjust = 0.0

        if args.mixup_minority is not None:
            model_args.mixup_minority = args.mixup_minority
        elif 'mixup_minority' in m_cfg:
            model_args.mixup_minority = m_cfg['mixup_minority']
        elif 'mixup_minority' in scenario_cfg:
            model_args.mixup_minority = scenario_cfg['mixup_minority']
        else:
            model_args.mixup_minority = 0.0

        model_args.use_tta = args.use_tta or m_cfg.get('use_tta', False) or scenario_cfg.get('use_tta', False)
        model_args.color_constancy = args.color_constancy or m_cfg.get('color_constancy', False) or scenario_cfg.get('color_constancy', False)

        print(f"\n⚙️  Configuring {m.upper()} from scenario '{args.scenario}': "
              f"epochs={model_args.epochs}, patience={model_args.patience}, batch_size={model_args.batch_size}, "
              f"lr1={model_args.lr_stage1}, lr2={model_args.lr_stage2}, mel_th={model_args.mel_threshold}, bcc_th={model_args.bcc_threshold}, "
              f"balanced_sampling={model_args.balanced_sampling}, logit_adjust={model_args.logit_adjust}, mixup={model_args.mixup_minority}, "
              f"tta={model_args.use_tta}, color_constancy={model_args.color_constancy}")

        result = train_single_model(m, model_args, output_dir, hw, train_df, ham_val_df, pad_val_df)
        all_results[m] = result

    # Summary Benchmark Output when benchmarking multiple models
    if len(all_results) > 1:
        summary_path = output_dir / 'summary_comparison.json'
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        plot_benchmark_summary(all_results, output_dir / 'benchmark_comparison.png')
        print_leaderboard(all_results)

    # Automatically Update Session Leaderboard and Global Archive Index
    update_session_leaderboard(session_dir)
    update_global_archive_index(Path('experiments'))


if __name__ == '__main__':
    main()
