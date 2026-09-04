"""
Unified CLI Entrypoint & Multi-Model Benchmark Runner.
Handles argument parsing, dynamic benchmark_scenarios.json configuration,
automatic session/scenario directory management, dataset preparation,
logging/prints, benchmark summarization, and archive indexing.
Delegates core training and modeling routines to the timm backbone engine (train_timm_models.py).
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from dataset import (
    build_split_manifest, ensure_pad_ufes20_download, load_pad_ufes20_validation,
    prepare_dataset, save_split_manifest, validate_image_paths
)
from train_timm_models import (
    MODEL_CONFIGS, configure_hardware_environment, train_single_model
)
from visualize import plot_benchmark_summary


@dataclass
class RunConfig:
    epochs: int
    patience: int
    batch_size: int
    lr_stage1: Optional[float]
    lr_stage2: Optional[float]
    seed: int
    img_size: Optional[int]
    mel_threshold: object
    bcc_threshold: object
    malignant_threshold: object
    balanced_sampling: bool
    logit_adjust: float
    mixup_alpha: float
    use_tta: bool
    color_constancy: bool
    stage1_epochs: int
    eval_precision: str
    no_cudnn: bool


def resolve_run_config(args, scenario_cfg: dict, model_cfg: dict) -> RunConfig:
    def value(name, default=None):
        cli_value = getattr(args, name, None)
        if cli_value is not None:
            return cli_value
        if name in model_cfg:
            return model_cfg[name]
        return scenario_cfg.get(name, default)

    return RunConfig(
        epochs=max(1, int(value('epochs', 20))),
        patience=max(1, int(value('patience', 4))),
        batch_size=int(value('batch_size', 32)),
        lr_stage1=value('lr_stage1'),
        lr_stage2=value('lr_stage2'),
        seed=int(value('seed', 42)),
        img_size=value('img_size'),
        mel_threshold=value('mel_threshold', 0.15),
        bcc_threshold=value('bcc_threshold', 'youden'),
        malignant_threshold=value('malignant_threshold'),
        balanced_sampling=bool(getattr(args, 'balanced_sampling', False) or model_cfg.get('balanced_sampling', scenario_cfg.get('balanced_sampling', False))),
        logit_adjust=float(value('logit_adjust', 0.0) or 0.0),
        mixup_alpha=float(value('mixup_alpha', value('mixup_minority', 0.0)) or 0.0),
        use_tta=bool(getattr(args, 'use_tta', False) or model_cfg.get('use_tta', scenario_cfg.get('use_tta', False))),
        color_constancy=bool(getattr(args, 'color_constancy', False) or model_cfg.get('color_constancy', scenario_cfg.get('color_constancy', False))),
        stage1_epochs=int(value('stage1_epochs', 3)),
        eval_precision=str(value('eval_precision', 'fp32')),
        no_cudnn=bool(getattr(args, 'no_cudnn', False)),
    )


def write_provenance(output_path: Path, config: RunConfig, hw: dict, split_manifest: dict):
    from importlib.metadata import PackageNotFoundError, version

    packages = {}
    for package in ('torch', 'torchvision', 'timm', 'numpy', 'pandas', 'scikit-learn'):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    try:
        git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        git_dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        git_sha, git_dirty = None, None
    serializable_hw = {key: str(value) if key in ('device', 'precision_dtype') else value for key, value in hw.items()}
    payload = {
        'config': asdict(config),
        'hardware': serializable_hw,
        'split_manifest_sha256': split_manifest['image_ids_sha256']['all'],
        'packages': packages,
        'python': sys.version,
        'platform': platform.platform(),
        'git_sha': git_sha,
        'git_dirty': git_dirty,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)


def load_benchmark_scenarios(scenarios_file: Union[str, Path] = "benchmark_scenarios.json") -> dict:
    """
    Loads benchmark scenarios and per-model hyperparameter presets dynamically from JSON.
    Supports both 'benchmark_scenarios.json' and 'benchmark_scenarions.json'.
    """
    target_path = Path(scenarios_file)

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
    best_m = max(all_results, key=lambda k: all_results[k].get('pad_mel_auc_roc', 0.0))
    print(f"🏆 Best Out-of-Domain Melanoma Model: {best_m.upper()} (PAD AUC: {all_results[best_m]['pad_mel_auc_roc']:.4f})\n")


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
                'ham_mel_auc_roc': round(float(data.get('ham_test_mel_auc_roc', data.get('ham_mel_auc_roc', 0.0))), 4),
                'ham_mel_auc_ci_low': round(float(data.get('ham_test_mel_auc_ci_low', 0.0)), 4),
                'ham_mel_auc_ci_high': round(float(data.get('ham_test_mel_auc_ci_high', 0.0)), 4),
                'mel_auc_roc': round(float(data.get('pad_mel_auc_roc', 0.0)), 4),
                'pad_mel_auc_ci_low': round(float(data.get('pad_mel_auc_ci_low', 0.0)), 4),
                'pad_mel_auc_ci_high': round(float(data.get('pad_mel_auc_ci_high', 0.0)), 4),
                'mel_auc_gap': round(float(data.get('mel_auc_gap', 0.0)), 4),
                'meets_auc_target_ham': bool(data.get('meets_auc_target_ham', False)),
                'meets_auc_target_pad': bool(data.get('meets_auc_target_pad', False)),
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
        df = df.sort_values(by=['mel_auc_roc', 'ham_mel_auc_roc'], ascending=[False, False])
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
                    h_auc = float(d.get('ham_test_mel_auc_roc', d.get('ham_mel_auc_roc', 0.0)))
                    p_auc = float(d.get('pad_mel_auc_roc', 0.0))
                    if h_auc > top_ham:
                        top_ham = h_auc
                    if p_auc > top_pad:
                        top_pad = p_auc
                except Exception:
                    pass

            ham_str = f"{top_ham:.4f}" if top_ham > 0 else "N/A"
            pad_str = f"{top_pad:.4f}" if top_pad > 0 else "N/A"
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
        "| Session Date / Folder | Runs Completed | Top HAM Test Mel AUC | Top PAD Mel AUC | Link |",
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
    parser.add_argument('--val-csv', type=str, default=None, help='In-domain tuning validation CSV (HAM10000)')
    parser.add_argument('--test-csv', type=str, default=None, help='Held-out in-domain test CSV (HAM10000)')
    parser.add_argument('--external-val-csv', type=str, default=None, help='Out-of-domain validation set CSV (PAD-UFES-20)')
    parser.add_argument('--cache-dir', type=str, default='./data_cache', help='Dataset cache directory')
    parser.add_argument('--prepared-dir', type=str, default='./dataset_treino', help='Prepared dataset directory')
    parser.add_argument('--pad-ufes-dir', type=str, default='./data_cache/pad_ufes_20_raw', help='PAD-UFES-20 dataset directory')
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
    parser.add_argument('--mixup-alpha', '--mixup-minority', dest='mixup_alpha', type=float, default=None,
                        help="Beta-distribution alpha parameter for Mixup data augmentation.")
    parser.add_argument('--stage1-epochs', type=int, default=None, help='Classifier-head warmup epochs')
    parser.add_argument('--no-cudnn', action='store_true', default=False, help='Disable cuDNN compatibility path')
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
    if base_seed != 42:
        raise SystemExit("This thesis pipeline uses the single fixed seed 42")
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)

    # Hardware Detection
    hw = configure_hardware_environment(disable_cudnn=args.no_cudnn)
    print_banner(hw, session_dir, args.scenario)

    # Load / Prepare Datasets
    data_cache = Path(args.cache_dir)
    prepared_dir = Path(args.prepared_dir)
    pad_ufes_dir = Path(args.pad_ufes_dir)

    # 1. Training, tuning validation, and held-out test (HAM10000)
    supplied_csvs = args.train_csv and args.val_csv and args.test_csv
    cached_manifest = None
    cached_csvs = all((session_dir / name).exists() for name in ('train_df.csv', 'ham_val_df.csv', 'ham_test_df.csv'))
    if supplied_csvs and all(Path(path).exists() for path in (args.train_csv, args.val_csv, args.test_csv)):
        train_df = pd.read_csv(args.train_csv)
        ham_val_df = pd.read_csv(args.val_csv)
        ham_test_df = pd.read_csv(args.test_csv)
    elif cached_csvs:
        manifest_path = session_dir / 'split_manifest.json'
        if not manifest_path.exists():
            raise RuntimeError(f"Cached splits require {manifest_path}")
        with open(manifest_path, 'r') as f:
            cached_manifest = json.load(f)
        if int(cached_manifest['random_state']) != base_seed:
            raise RuntimeError(f"Cached split seed {cached_manifest['random_state']} does not match requested seed {base_seed}")
        train_df = pd.read_csv(session_dir / 'train_df.csv')
        ham_val_df = pd.read_csv(session_dir / 'ham_val_df.csv')
        ham_test_df = pd.read_csv(session_dir / 'ham_test_df.csv')
    else:
        print("[dataset] Preparing HAM10000 dataset (70/10/20 lesion-grouped split)...")
        train_df, ham_val_df, ham_test_df = prepare_dataset(
            data_cache, prepared_dir, random_state=base_seed, oversample=False
        )
    invalid_images = {}
    cleaned_splits = []
    for split_name, frame in (('train', train_df), ('val', ham_val_df), ('test', ham_test_df)):
        clean_frame, bad_paths = validate_image_paths(frame)
        invalid_images[split_name] = bad_paths
        cleaned_splits.append(clean_frame)
    train_df, ham_val_df, ham_test_df = cleaned_splits
    with open(session_dir / 'invalid_images.json', 'w') as f:
        json.dump(invalid_images, f, indent=2)
    train_df.to_csv(session_dir / 'train_df.csv', index=False)
    ham_val_df.to_csv(session_dir / 'ham_val_df.csv', index=False)
    ham_test_df.to_csv(session_dir / 'ham_test_df.csv', index=False)

    split_manifest = build_split_manifest(train_df, ham_val_df, ham_test_df, random_state=base_seed)
    if cached_manifest and cached_manifest['image_ids_sha256']['all'] != split_manifest['image_ids_sha256']['all']:
        raise RuntimeError("Cached split contents do not match split_manifest.json")
    save_split_manifest(split_manifest, session_dir / 'split_manifest.json')
    original_priors = {
        class_name: float((train_df['dx'] == class_name).mean())
        for class_name in sorted(train_df['dx'].unique())
    }
    with open(session_dir / 'class_priors.json', 'w') as f:
        json.dump(original_priors, f, indent=2)

    # 2. Out-of-Domain Validation (PAD-UFES-20)
    if args.external_val_csv and Path(args.external_val_csv).exists():
        pad_val_df = pd.read_csv(args.external_val_csv)
    elif (session_dir / 'pad_val_df.csv').exists():
        pad_val_df = pd.read_csv(session_dir / 'pad_val_df.csv')
    else:
        print("[dataset] Preparing PAD-UFES-20 validation dataset...")
        pad_dir = ensure_pad_ufes20_download(data_cache) if not pad_ufes_dir.exists() else pad_ufes_dir
        pad_val_df = load_pad_ufes20_validation(pad_dir, session_dir / 'pad_label_mapping.json')
    pad_val_df, invalid_pad = validate_image_paths(pad_val_df)
    pad_val_df.to_csv(session_dir / 'pad_val_df.csv', index=False)
    with open(session_dir / 'invalid_pad_images.json', 'w') as f:
        json.dump(invalid_pad, f, indent=2)

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
        m_cfg = scenario_models_cfg.get(m, {})
        resolved_cfg = resolve_run_config(args, scenario_cfg, m_cfg)
        if resolved_cfg.seed != base_seed:
            raise RuntimeError(f"Model seed {resolved_cfg.seed} does not match split seed {base_seed}")
        model_args = argparse.Namespace(**asdict(resolved_cfg))

        write_provenance(output_dir / m / 'config.json', resolved_cfg, hw, split_manifest)

        print(f"\n⚙️  Configuring {m.upper()} from scenario '{args.scenario}': "
              f"epochs={model_args.epochs}, patience={model_args.patience}, batch_size={model_args.batch_size}, "
              f"lr1={model_args.lr_stage1}, lr2={model_args.lr_stage2}, mel_th={model_args.mel_threshold}, bcc_th={model_args.bcc_threshold}, "
              f"balanced_sampling={model_args.balanced_sampling}, logit_adjust={model_args.logit_adjust}, mixup={model_args.mixup_alpha}, "
              f"tta={model_args.use_tta}, color_constancy={model_args.color_constancy}")

        result = train_single_model(
            m, model_args, output_dir, hw, train_df, ham_val_df, ham_test_df, pad_val_df
        )
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
