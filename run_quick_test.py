#!/usr/bin/env python3
"""
Quick 5-Epoch Pipeline Verification Runner for MobileNet V3, V4, and V5.
Session ID: 19_08_2025_test_run_check_pipelines
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd

import torch
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Session Configuration
SESSION_NAME = "19_08_2025_test_run_check_pipelines"
EXPERIMENTS_DIR = Path("experiments")
SESSION_DIR = EXPERIMENTS_DIR / SESSION_NAME
MODELS = ["v3", "v4", "v5"]
EPOCHS = 5
PATIENCE = 5
BATCH_SIZE = 32
PYTHON_BIN = sys.executable

class TeeLogger:
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_path, 'a', buffering=1, encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

def main():
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SESSION_DIR / "execution.log"
    sys.stdout = TeeLogger(log_path)
    sys.stderr = sys.stdout

    print(f"\n{'='*85}")
    print(f" 🚀 Starting Pipeline Test Run Check (5 Epochs)")
    print(f" Session ID: {SESSION_NAME}")
    print(f" Target Models: {', '.join(MODELS)}")
    print(f" Output Directory: {SESSION_DIR.resolve()}")
    print(f" Log File: {log_path.resolve()}")
    print(f"{'='*85}\n")

    # 1. Dataset verification & copying
    train_csv = SESSION_DIR / "train_df.csv"
    ham_val_csv = SESSION_DIR / "ham_val_df.csv"
    pad_val_csv = SESSION_DIR / "pad_val_df.csv"

    data_cache = Path("./data_cache")
    prepared_dir = Path("./dataset_treino")

    if not (train_csv.exists() and ham_val_csv.exists() and pad_val_csv.exists()):
        # Check parent experiments or mobilenet_outputs first
        cached_train = Path("experiments/18_08_2026_run2/train_df.csv")
        cached_ham = Path("experiments/18_08_2026_run2/ham_val_df.csv")
        cached_pad = Path("experiments/18_08_2026_run2/pad_val_df.csv")

        if cached_train.exists() and cached_ham.exists() and cached_pad.exists():
            print("  [Setup] Copying cached dataset splits...")
            pd.read_csv(cached_train).to_csv(train_csv, index=False)
            pd.read_csv(cached_ham).to_csv(ham_val_csv, index=False)
            pd.read_csv(cached_pad).to_csv(pad_val_csv, index=False)
        else:
            print("  [Setup] Generating dataset splits from scratch...")
            from dataset import prepare_dataset, ensure_pad_ufes20_download, load_pad_ufes20_validation
            t_df, h_df = prepare_dataset(cache_root=data_cache, prepared_dir=prepared_dir, random_state=42)
            pad_dir = ensure_pad_ufes20_download(data_cache)
            p_df = load_pad_ufes20_validation(pad_dir)
            t_df.to_csv(train_csv, index=False)
            h_df.to_csv(ham_val_csv, index=False)
            p_df.to_csv(pad_val_csv, index=False)

    print(f"  ✓ Datasets Ready: Train={len(pd.read_csv(train_csv))}, HAM_Val={len(pd.read_csv(ham_val_csv))}, PAD_Val={len(pd.read_csv(pad_val_csv))}\n")

    # 2. Execute models sequentially
    results_list = []
    for idx, model in enumerate(MODELS, start=1):
        print(f"\n{'#'*85}")
        print(f" ⚙️ [{idx}/{len(MODELS)}] Running 5-Epoch Verification: {model.upper()}")
        print(f"{'#'*85}\n")

        cmd = [
            PYTHON_BIN, "-u", "train_timm_models.py",
            "--model", model,
            "--epochs", str(EPOCHS),
            "--batch-size", str(BATCH_SIZE),
            "--patience", str(PATIENCE),
            "--train-csv", str(train_csv),
            "--val-csv", str(ham_val_csv),
            "--external-val-csv", str(pad_val_csv),
            "--val-dataset", "both",
            "--output-dir", str(SESSION_DIR),
            "--seed", "42"
        ]

        t0 = time.time()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()

        process.wait()
        elapsed = time.time() - t0

        if process.returncode == 0:
            print(f"\n  ✓ Successfully finished {model.upper()} in {elapsed:.1f}s")
            res_file = SESSION_DIR / model / "results.json"
            if res_file.exists():
                with open(res_file) as f:
                    r_data = json.load(f)
                
                entry = {
                    "model": model.upper(),
                    "epochs": EPOCHS,
                    "ham_accuracy": r_data.get("ham_accuracy", 0.0),
                    "pad_accuracy": r_data.get("pad_accuracy", r_data.get("accuracy", 0.0)),
                    "domain_gap": r_data.get("domain_gap", r_data.get("domain_gap_drop", 0.0)),
                    "ham_mel_recall": r_data.get("ham_mel_recall", 0.0),
                    "pad_mel_recall": r_data.get("pad_mel_recall", r_data.get("mel_recall", 0.0)),
                    "pad_bcc_recall": r_data.get("pad_bcc_recall", r_data.get("bcc_recall", 0.0)),
                    "elapsed_sec": round(elapsed, 1),
                    "status": "SUCCESS"
                }
                results_list.append(entry)
        else:
            print(f"\n  ✗ Failed to run {model.upper()} (returncode: {process.returncode})")

    # 3. Create Leaderboard and Summary
    if results_list:
        df_results = pd.DataFrame(results_list)
        df_results.to_csv(SESSION_DIR / "master_leaderboard.csv", index=False)

        summary_path = SESSION_DIR / "SUMMARY.md"
        with open(summary_path, "w") as f:
            f.write(f"# 🧪 5-Epoch Verification Benchmark: `{SESSION_NAME}`\n\n")
            f.write(f"*Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            f.write("### 📊 Dual-Domain Model Verification Leaderboard\n\n")
            f.write(df_results.to_markdown(index=False))
            f.write("\n\n---\n")
            f.write("### 🖼️ Generated Visualizations per Model:\n\n")
            for m in MODELS:
                f.write(f"- **{m.upper()}**:\n")
                f.write(f"  - In-Domain Heatmaps: `experiments/{SESSION_NAME}/{m}/ham10000/gradcam_heatmaps.png`\n")
                f.write(f"  - Out-of-Domain Heatmaps: `experiments/{SESSION_NAME}/{m}/pad_ufes_20/gradcam_heatmaps.png`\n")
                f.write(f"  - Domain Shift Diagnostic: `experiments/{SESSION_NAME}/{m}/domain_comparison.png`\n")
                f.write(f"  - Training Curves: `experiments/{SESSION_NAME}/{m}/training_curves.png`\n")

        print(f"\n{'='*85}")
        print(f" 🏁 Verification Benchmark Finished Successfully!")
        print(f" Leaderboard & Summary saved to: {summary_path.resolve()}")
        print(f"{'='*85}\n")

    # Update Global Archive Index
    from run_scenarios import update_global_archive_index
    update_global_archive_index(EXPERIMENTS_DIR)

if __name__ == "__main__":
    main()
