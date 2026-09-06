#!/usr/bin/env python3
"""
Multi-Model Ensemble Engine for Skin Lesion Classification Benchmarks.
Combines multiple MobileNet checkpoints (V1 to V5) using soft probability voting,
evaluates dual-domain performance (In-Domain HAM10000 and Out-of-Domain PAD-UFES-20),
and computes calibrated clinical triage operating points.
"""

from contextlib import nullcontext
import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

from dataset import (
    NUM_CLASSES,
    CLASS_NAMES,
    prepare_dataset,
    load_pad_ufes20_validation,
    ensure_pad_ufes20_download
)
from metrics import select_logit_adjust
from train_timm_models import (
    MODEL_CONFIGS,
    SkinDataset,
    _evaluate_binary_triage,
    build_transforms,
    plot_confusion_matrices,
    plot_decision_confusion_matrices,
    plot_per_class_metrics,
    plot_roc_curves
)


def load_model_checkpoint(model_name: str, checkpoint_path: Path, device: torch.device):
    """Initializes model architecture and loads weights from checkpoint."""
    import timm
    cfg = MODEL_CONFIGS[model_name]
    timm_name = cfg['timm_name']
    model = timm.create_model(timm_name, pretrained=False, num_classes=NUM_CLASSES)
    state = torch.load(checkpoint_path, map_location=device)
    # Handle possible module. prefix from DataParallel
    cleaned_state = {}
    for k, v in state.items():
        clean_k = k.replace('module.', '') if k.startswith('module.') else k
        cleaned_state[clean_k] = v
    model.load_state_dict(cleaned_state)
    model = model.to(device)
    model.eval()
    return model, cfg


def predict_loader(model, loader, device, precision_dtype, has_cuda, use_tta=False):
    """Computes softmax probabilities for all samples in loader with optional TTA."""
    all_probs = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            views = [x, torch.flip(x, dims=[-1]), torch.flip(x, dims=[-2])] if use_tta else [x]
            with nullcontext():
                outputs = [model(view) for view in views]
            probs = torch.stack([torch.softmax(output.float(), dim=1) for output in outputs]).mean(dim=0)
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_probs)


def evaluate_ensemble(
    models_dict: dict,
    loaders: dict,
    targets: np.ndarray,
    device: torch.device,
    precision_dtype,
    has_cuda: bool,
    weights: dict = None,
    mel_threshold='sens90',
    bcc_threshold='sens90',
    malignant_threshold='sens90',
    use_tta=False
) -> dict:
    """Computes weighted probability ensemble and comprehensive clinical metrics."""
    model_names = list(models_dict.keys())
    if weights is None:
        weights = {m: 1.0 / len(model_names) for m in model_names}
    else:
        # Normalize weights
        w_sum = sum(weights.values())
        weights = {m: w / w_sum for m, w in weights.items()}

    print(f"  🤝 Combining {len(models_dict)} models with weights: {weights}")
    
    ensemble_probs = np.zeros((len(targets), NUM_CLASSES), dtype=np.float32)
    per_model_probs = {}

    for m_name, model in models_dict.items():
        w = weights[m_name]
        print(f"    ↳ Evaluating model '{m_name}' (weight: {w:.3f})...")
        probs = predict_loader(model, loaders[m_name], device, precision_dtype, has_cuda, use_tta=use_tta)
        per_model_probs[m_name] = probs
        ensemble_probs += w * probs

    all_preds = np.argmax(ensemble_probs, axis=1)
    acc = float(np.mean(all_preds == targets))
    report = classification_report(targets, all_preds, labels=range(NUM_CLASSES), target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    # Melanoma continuous AUC-ROC
    mel_idx = CLASS_NAMES.index('mel')
    bin_mel = (targets == mel_idx).astype(int)
    mel_probs = ensemble_probs[:, mel_idx]
    mel_auc = float(roc_auc_score(bin_mel, mel_probs)) if len(np.unique(bin_mel)) > 1 else 0.0

    # Macro AUC-ROC
    valid_aucs = []
    for c_idx, c_name in enumerate(CLASS_NAMES):
        bin_c = (targets == c_idx).astype(int)
        if len(np.unique(bin_c)) > 1:
            valid_aucs.append(float(roc_auc_score(bin_c, ensemble_probs[:, c_idx])))
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0

    # Triage screening calculations
    mel_res = _evaluate_binary_triage(mel_probs, bin_mel, threshold_spec=mel_threshold, default_th=0.15)

    bcc_idx = CLASS_NAMES.index('bcc')
    bin_bcc = (targets == bcc_idx).astype(int)
    bcc_probs = ensemble_probs[:, bcc_idx]
    bcc_res = _evaluate_binary_triage(bcc_probs, bin_bcc, threshold_spec=bcc_threshold, default_th=0.15)

    mal_indices = [CLASS_NAMES.index(c) for c in ['mel', 'bcc', 'akiec']]
    bin_mal = np.isin(targets, mal_indices).astype(int)
    mal_probs = np.sum(ensemble_probs[:, mal_indices], axis=1)
    mal_res = _evaluate_binary_triage(mal_probs, bin_mal, threshold_spec=malignant_threshold or 'sens90', default_th=0.20)

    return {
        'accuracy': acc,
        'weighted_avg_f1': float(report.get('weighted avg', {}).get('f1-score', 0.0)),
        'macro_avg_f1': float(report.get('macro avg', {}).get('f1-score', 0.0)),
        'mel_recall': float(report.get('mel', {}).get('recall', 0.0)),
        'bcc_recall': float(report.get('bcc', {}).get('recall', 0.0)),
        'mel_auc_roc': mel_auc,
        'macro_auc_roc': macro_auc,
        'mel_triage_threshold': mel_res['threshold'],
        'mel_triage_recall': mel_res['sensitivity'],
        'mel_triage_spec': mel_res['specificity'],
        'mel_triage_detected': mel_res['detected'],
        'mel_operating_points': mel_res['operating_points'],
        'bcc_triage_threshold': bcc_res['threshold'],
        'bcc_triage_recall': bcc_res['sensitivity'],
        'bcc_triage_spec': bcc_res['specificity'],
        'bcc_triage_detected': bcc_res['detected'],
        'malignant_triage_threshold': mal_res['threshold'],
        'malignant_triage_recall': mal_res['sensitivity'],
        'malignant_triage_spec': mal_res['specificity'],
        'malignant_triage_detected': mal_res['detected'],
        'all_preds': all_preds,
        'all_targets': targets,
        'all_probs': ensemble_probs,
        'report': report
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Model Ensemble for Skin Lesion Classification')
    parser.add_argument('--session-dir', type=str, required=True, help='Path to session directory with trained models')
    parser.add_argument('--models', nargs='+', default=['v2', 'v4', 'v5'], help='List of model names to ensemble')
    parser.add_argument('--weights', nargs='+', type=float, default=None, help='Weights for each model (must match length of --models)')
    parser.add_argument('--use-tta', action='store_true', default=False, help='Apply 3-view probability-averaged Test-Time Augmentation')
    parser.add_argument('--color-constancy', action='store_true', default=False, help='Apply Shades-of-Gray transform')
    parser.add_argument('--batch-size', type=int, default=32, help='Inference batch size')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for ensemble artifacts')
    return parser.parse_args()


def main():
    args = parse_args()
    session_dir = Path(args.session_dir)
    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else session_dir / 'ensemble'
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    has_bf16 = (device.type == 'cuda' and torch.cuda.is_bf16_supported())
    precision_dtype = torch.bfloat16 if has_bf16 else torch.float32

    # Load validation sets
    ham_csv = session_dir / 'ham_val_df.csv'
    ham_test_csv = session_dir / 'ham_test_df.csv'
    pad_csv = session_dir / 'pad_val_df.csv'
    if not all(path.exists() for path in (ham_csv, ham_test_csv, pad_csv)):
        raise FileNotFoundError(f"HAM validation/test and PAD CSVs are required in {session_dir}")

    ham_val_df = pd.read_csv(ham_csv)
    ham_test_df = pd.read_csv(ham_test_csv)
    pad_val_df = pd.read_csv(pad_csv)

    print("=" * 80)
    print(" 🤝 Skin Lesion Multi-Model Soft Voting Ensemble Engine")
    print(f" 📂 Session Dir: {session_dir}")
    print(f" 🎯 Ensembling Models: {args.models}")
    print(f" 🖼️ Datasets: HAM10000 ({len(ham_val_df)}) | PAD-UFES-20 ({len(pad_val_df)})")
    print("=" * 80)

    # Locate and load checkpoints
    loaded_models = {}
    for m in args.models:
        candidate_paths = [
            session_dir / m / 'best_model.pth',
            session_dir / 'scenarios' / 'standard' / m / 'best_model.pth',
            session_dir / 'scenarios' / 'medium' / m / 'best_model.pth',
            session_dir / 'scenarios' / 'low' / m / 'best_model.pth',
        ]
        found = next((p for p in candidate_paths if p.exists()), None)
        if found:
            print(f"  📦 Loading checkpoint for {m.upper()} from {found}")
            mod, _ = load_model_checkpoint(m, found, device)
            loaded_models[m] = mod
        else:
            print(f"  ⚠️ Warning: Checkpoint for {m} not found in {session_dir}. Skipping.")

    if not loaded_models:
        raise RuntimeError("No model checkpoints could be loaded!")

    ham_loaders, ham_test_loaders, pad_loaders = {}, {}, {}
    for model_name, model in loaded_models.items():
        img_size = MODEL_CONFIGS[model_name]['input_size']
        val_transform, _ = build_transforms(model, img_size, train=False, color_constancy=args.color_constancy)
        loader_kwargs = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 2}
        ham_loaders[model_name] = DataLoader(SkinDataset(ham_val_df, transform=val_transform), **loader_kwargs)
        ham_test_loaders[model_name] = DataLoader(SkinDataset(ham_test_df, transform=val_transform), **loader_kwargs)
        pad_loaders[model_name] = DataLoader(SkinDataset(pad_val_df, transform=val_transform), **loader_kwargs)

    weights_dict = None
    if args.weights:
        weights_dict = {m: w for m, w in zip(loaded_models.keys(), args.weights[:len(loaded_models)])}

    # 1. Evaluate on In-Domain HAM10000 to calibrate thresholds
    print("\n--- Step 1: In-Domain Evaluation & Calibration (HAM10000) ---")
    ham_targets = np.array([CLASS_NAMES.index(c) for c in ham_val_df['dx']])
    ham_results = evaluate_ensemble(
        loaded_models, ham_loaders, ham_targets, device, precision_dtype,
        has_cuda=(device.type == 'cuda'), weights=weights_dict,
        mel_threshold='youden', bcc_threshold='youden', use_tta=args.use_tta
    )
    calibrated_mel_th = ham_results['mel_triage_threshold']
    calibrated_bcc_th = ham_results['bcc_triage_threshold']
    calibrated_mal_th = ham_results['malignant_triage_threshold']
    prior_path = session_dir / 'class_priors.json'
    if prior_path.exists():
        with open(prior_path) as f:
            prior_map = json.load(f)
        class_priors = np.asarray([prior_map[name] for name in CLASS_NAMES], dtype=float)
    else:
        class_priors = np.bincount(ham_targets, minlength=NUM_CLASSES).astype(float)
        class_priors /= class_priors.sum()
    selected_tau, _ = select_logit_adjust(ham_results['all_probs'], ham_results['all_targets'], class_priors)
    ham_test_targets = np.array([CLASS_NAMES.index(c) for c in ham_test_df['dx']])
    ham_test_results = evaluate_ensemble(
        loaded_models, ham_test_loaders, ham_test_targets, device, precision_dtype,
        has_cuda=(device.type == 'cuda'), weights=weights_dict,
        mel_threshold=calibrated_mel_th, bcc_threshold=calibrated_bcc_th,
        malignant_threshold=calibrated_mal_th, use_tta=args.use_tta
    )

    # 2. Evaluate on Out-of-Domain PAD-UFES-20 using calibrated thresholds
    print("\n--- Step 2: Out-of-Domain Clinical Triage (PAD-UFES-20) ---")
    pad_targets = np.array([CLASS_NAMES.index(c) for c in pad_val_df['dx']])
    pad_results = evaluate_ensemble(
        loaded_models, pad_loaders, pad_targets, device, precision_dtype,
        has_cuda=(device.type == 'cuda'), weights=weights_dict,
        mel_threshold=calibrated_mel_th, bcc_threshold=calibrated_bcc_th,
        malignant_threshold=calibrated_mal_th, use_tta=args.use_tta
    )

    domain_gap = ham_test_results['accuracy'] - pad_results['accuracy']

    summary = {
        'models': list(loaded_models.keys()),
        'weights': weights_dict or {m: 1.0 / len(loaded_models) for m in loaded_models},
        'use_tta': args.use_tta,
        'color_constancy': args.color_constancy,
        'ham_val_mel_auc_roc': ham_results['mel_auc_roc'],
        'ham_accuracy': ham_test_results['accuracy'],
        'ham_macro_auc_roc': ham_test_results['macro_auc_roc'],
        'ham_mel_auc_roc': ham_test_results['mel_auc_roc'],
        'ham_mel_triage_recall': ham_test_results['mel_triage_recall'],
        'pad_accuracy': pad_results['accuracy'],
        'pad_macro_auc_roc': pad_results['macro_auc_roc'],
        'pad_mel_auc_roc': pad_results['mel_auc_roc'],
        'domain_gap': domain_gap,
        'mel_triage_threshold': calibrated_mel_th,
        'pad_mel_triage_recall': pad_results['mel_triage_recall'],
        'pad_mel_triage_spec': pad_results['mel_triage_spec'],
        'pad_mel_triage_detected': pad_results['mel_triage_detected'],
        'bcc_triage_threshold': calibrated_bcc_th,
        'pad_bcc_triage_recall': pad_results['bcc_triage_recall'],
        'pad_bcc_triage_spec': pad_results['bcc_triage_spec'],
        'pad_bcc_triage_detected': pad_results['bcc_triage_detected'],
        'malignant_triage_threshold': calibrated_mal_th,
        'selected_logit_adjust': selected_tau,
        'logit_adjust_source': 'ham_val_balanced_accuracy',
        'pad_malignant_triage_recall': pad_results['malignant_triage_recall'],
        'pad_malignant_triage_spec': pad_results['malignant_triage_spec'],
        'pad_malignant_triage_detected': pad_results['malignant_triage_detected']
    }

    with open(output_dir / 'ensemble_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    plot_confusion_matrices(pad_results['all_targets'], pad_results['all_preds'], CLASS_NAMES, output_dir / 'pad_confusion_matrix.png', model_name="Ensemble (PAD-UFES-20)")
    plot_decision_confusion_matrices(
        ham_test_results['all_probs'], ham_test_results['all_targets'], CLASS_NAMES, class_priors,
        selected_tau, calibrated_mal_th, output_dir / 'ham_confusion_matrix_decision.png', model_name="Ensemble (HAM10000 Test)"
    )
    plot_decision_confusion_matrices(
        pad_results['all_probs'], pad_results['all_targets'], CLASS_NAMES, class_priors,
        selected_tau, calibrated_mal_th, output_dir / 'pad_confusion_matrix_decision.png', model_name="Ensemble (PAD-UFES-20)"
    )
    plot_roc_curves(pad_results['all_targets'], pad_results['all_probs'], CLASS_NAMES, output_dir / 'pad_roc_curves.png', model_name="Ensemble (PAD-UFES-20)")

    print("\n" + "=" * 80)
    print(" 🏆 ENSEMBLE BENCHMARK RESULTS:")
    print(f"   In-Domain (HAM10000 Test): Acc={ham_test_results['accuracy']:.2%} | Macro AUC={ham_test_results['macro_auc_roc']:.4f} | Mel AUC={ham_test_results['mel_auc_roc']:.4f}")
    print(f"   Out-of-Domain (PAD-UFES): Acc={pad_results['accuracy']:.2%} | Macro AUC={pad_results['macro_auc_roc']:.4f} | Mel AUC={pad_results['mel_auc_roc']:.4f}")
    print(f"     ↳ MEL Triage (tau={calibrated_mel_th:.2f}): Recall={pad_results['mel_triage_recall']:.2%} | Spec={pad_results['mel_triage_spec']:.2%} | Detected={pad_results['mel_triage_detected']}")
    print(f"     ↳ BCC Triage (tau={calibrated_bcc_th:.2f}): Recall={pad_results['bcc_triage_recall']:.2%} | Spec={pad_results['bcc_triage_spec']:.2%} | Detected={pad_results['bcc_triage_detected']}")
    print(f"     ↳ Malignancy Screen:    Recall={pad_results['malignant_triage_recall']:.2%} | Spec={pad_results['malignant_triage_spec']:.2%} | Detected={pad_results['malignant_triage_detected']}")
    print(f"   Clinical Domain Gap:      -{domain_gap*100:.2f}%")
    print(f"   Results saved to:         {output_dir / 'ensemble_results.json'}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
