"""
PyTorch & timm Pretrained Models Trainer (MobileNet V1 - V5)
Dual-Domain Multi-Phase Evaluation Pipeline:
  - In-Domain Evaluation: HAM10000 (Dermoscopy)
  - Out-of-Domain Evaluation: PAD-UFES-20 (Clinical Smartphone)
  - Full tracking across Stage 1 (Warmup) and Stage 2 (Fine-Tuning)
"""

import argparse
from contextlib import nullcontext
import hashlib
import os
import sys
import json
import random
import time
from functools import partial
from time import perf_counter
from pathlib import Path


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
from timm.data import resolve_data_config
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc

from dataset import CLASS_NAMES, NUM_CLASSES
from metrics import (
    _evaluate_binary_triage, bootstrap_metric_ci, confusion_summary, decide_argmax,
    decide_malignant_gated, decide_prior_corrected, expected_calibration_error, fit_temperature,
    meets_auc_target, restricted_class_accuracy, select_logit_adjust,
)
from visualize import (
    plot_training_curves, plot_confusion_matrices, plot_decision_confusion_matrices,
    plot_per_class_metrics, generate_gradcam_gallery, plot_domain_comparison,
    plot_roc_curves, plot_dual_roc_comparison, plot_reliability_diagram
)


# ─── Modular Hardware Engine ────────────────────────────────────────────────

def configure_hardware_environment(disable_cudnn: bool = False) -> dict:
    """Detects GPU architecture, compute capability, VRAM size, native BF16/FP16 support."""
    cudnn_enabled = False
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        device_count = torch.cuda.device_count()
        has_bf16 = torch.cuda.is_bf16_supported()
        precision_dtype = torch.bfloat16 if has_bf16 else torch.float16
        precision_name = 'BFloat16 (BF16)' if has_bf16 else 'Float16 (FP16)'
        disable_cudnn = disable_cudnn or os.environ.get('TCC_DISABLE_CUDNN') == '1'
        torch.backends.cudnn.enabled = not disable_cudnn
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if torch.backends.cudnn.enabled:
            try:
                test_x = torch.randn(1, 1, 4, 4, device=device)
                test_conv = nn.Conv2d(1, 1, 2).to(device)
                _ = test_conv(test_x)
                torch.cuda.synchronize()
                cudnn_enabled = True
            except Exception:
                torch.backends.cudnn.enabled = False

    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        device_name = 'Apple Silicon (MPS)'
        total_vram_gb = 16.0
        device_count = 1
        has_bf16 = False
        precision_dtype = torch.float32
        precision_name = 'Float32 (MPS)'
    else:
        device = torch.device('cpu')
        device_name = 'CPU'
        total_vram_gb = 0.0
        device_count = 1
        has_bf16 = False
        precision_dtype = torch.float32
        precision_name = 'Float32 (CPU)'

    return {
        'device': device,
        'device_name': device_name,
        'vram_gb': total_vram_gb,
        'device_count': device_count,
        'has_bf16': has_bf16,
        'precision_dtype': precision_dtype,
        'precision_name': precision_name,
        'cudnn_enabled': cudnn_enabled,
    }


def compute_adaptive_batch_strategy(vram_gb: float, model_name: str, requested_batch: int = 32) -> dict:
    """Computes physical micro-batch size and gradient accumulation steps."""
    is_large_model = model_name in ('v5', 'v4convl')

    if vram_gb >= 23.5:
        tier_gb, micro_batch = 24, 16 if is_large_model else requested_batch
    elif vram_gb >= 11.5:
        tier_gb, micro_batch = 12, 8 if is_large_model else min(requested_batch, 32)
    elif vram_gb >= 7.5:
        tier_gb, micro_batch = 8, 4 if is_large_model else min(requested_batch, 16)
    elif vram_gb >= 3.5:
        tier_gb, micro_batch = 4, 2 if is_large_model else min(requested_batch, 8)
    else:
        tier_gb, micro_batch = 0, 1 if is_large_model else min(requested_batch, 4)

    grad_accum_steps = max(1, int(np.ceil(requested_batch / micro_batch)))
    return {'micro_batch': micro_batch, 'grad_accum_steps': grad_accum_steps, 'tier_gb': tier_gb}




MODEL_CONFIGS = {
    'v1':      {'timm_name': 'mobilenetv1_100',                        'input_size': 224, 'default_lr1': 1e-3, 'default_lr2': 1e-4, 'weight_decay': 1e-4, 'pretrained': 'ImageNet-1k (timm)'},
    'v2':      {'timm_name': 'mobilenetv2_100',                        'input_size': 224, 'default_lr1': 1e-3, 'default_lr2': 1e-4, 'weight_decay': 1e-4, 'pretrained': 'ImageNet-1k (timm)'},
    'v3':      {'timm_name': 'mobilenetv3_large_100',                  'input_size': 224, 'default_lr1': 1e-3, 'default_lr2': 1e-4, 'weight_decay': 1e-4, 'pretrained': 'ImageNet-1k (timm)'},
    'v3large': {'timm_name': 'mobilenetv3_large_100',                  'input_size': 224, 'default_lr1': 1e-3, 'default_lr2': 1e-4, 'weight_decay': 1e-4, 'pretrained': 'ImageNet-1k (timm)'},
    'v3small': {'timm_name': 'mobilenetv3_small_100',                  'input_size': 224, 'default_lr1': 1e-3, 'default_lr2': 1e-4, 'weight_decay': 1e-4, 'pretrained': 'ImageNet-1k (timm)'},
    'v4':      {'timm_name': 'mobilenetv4_conv_medium.e500_r256_in1k', 'input_size': 256, 'default_lr1': 1e-3, 'default_lr2': 5e-5, 'weight_decay': 1e-4, 'pretrained': 'ImageNet-1k (Hugging Face / timm)'},
    'v4conv':  {'timm_name': 'mobilenetv4_conv_medium.e500_r256_in1k', 'input_size': 256, 'default_lr1': 1e-3, 'default_lr2': 5e-5, 'weight_decay': 1e-4, 'pretrained': 'ImageNet-1k (Hugging Face / timm)'},
    'v4convl': {'timm_name': 'mobilenetv4_conv_large.e500_r384_in1k',  'input_size': 384, 'default_lr1': 1e-3, 'default_lr2': 5e-5, 'weight_decay': 1e-4, 'pretrained': 'ImageNet-1k (Hugging Face / timm)'},
    'v5':      {'timm_name': 'mobilenetv5_300m.gemma3n',               'input_size': 256, 'default_lr1': 5e-4, 'default_lr2': 2e-5, 'weight_decay': 5e-4, 'pretrained': 'Google Gemma3n Vision (Hugging Face / timm)'},
}


class SkinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.paths = df['path'].values
        self.labels = [CLASS_NAMES.index(c) for c in df['dx'].values]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


class ShadesOfGray(object):
    """Applies Minkowski p-norm (Shades-of-Gray) color constancy to an image tensor.
    Standardizes illumination differences between polarized dermoscopy (HAM10000)
    and ambient/flash clinical smartphone photography (PAD-UFES-20).
    """
    def __init__(self, p: float = 6.0, eps: float = 1e-6):
        self.p = float(p)
        self.eps = float(eps)

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(img_tensor, torch.Tensor):
            return img_tensor
        img = img_tensor.float().clamp(min=0.0, max=1.0)
        p_pow = torch.pow(img.clamp(min=self.eps), self.p)
        ill = torch.pow(torch.mean(p_pow, dim=(1, 2)), 1.0 / self.p)
        norm_factor = torch.norm(ill, p=2) + self.eps
        scale = (ill * float(np.sqrt(3.0)) / norm_factor).view(3, 1, 1)
        normalized = img / (scale + self.eps)
        return normalized.clamp(0.0, 1.0)


_MIXUP_RNG = np.random.default_rng(42)


def set_seed(seed: int):
    global _MIXUP_RNG
    random.seed(seed)
    np.random.seed(seed)
    _MIXUP_RNG = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int, seed: int):
    worker_seed = seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_transforms(model, img_size: int, train: bool, color_constancy: bool = False):
    data_cfg = resolve_data_config({}, model=model)
    interpolation = data_cfg.get('interpolation', 'bilinear')
    if train:
        transform_list = [
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode(interpolation)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
            transforms.RandomAutocontrast(p=0.3),
            transforms.ToTensor(),
        ]
    else:
        transform_list = [
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode(interpolation)),
            transforms.ToTensor(),
        ]
    if color_constancy:
        transform_list.append(ShadesOfGray(p=6.0))
    transform_list.append(transforms.Normalize(mean=data_cfg['mean'], std=data_cfg['std']))
    return transforms.Compose(transform_list), {
        'input_size': img_size,
        'mean': list(data_cfg['mean']),
        'std': list(data_cfg['std']),
        'interpolation': interpolation,
    }


class PyTorchFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = inputs.float()
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-torch.clamp(ce_loss, max=15.0))
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()


def compute_class_weights(labels_series: pd.Series) -> dict:
    counts = labels_series.value_counts()
    n_samples = len(labels_series)
    weights = {}
    for idx, c in enumerate(CLASS_NAMES):
        count = counts.get(c, 0)
        weights[idx] = (n_samples / (NUM_CLASSES * count)) if count > 0 else 1.0
    w_sum = sum(weights.values())
    return {k: (v / w_sum) * NUM_CLASSES for k, v in weights.items()}


def compute_class_priors(labels_series: pd.Series) -> np.ndarray:
    """Computes empirical training set class prior distribution vector."""
    counts = labels_series.value_counts()
    n_samples = len(labels_series)
    return np.array([counts.get(c, 0) / max(n_samples, 1) for c in CLASS_NAMES], dtype=np.float32)


def mixup_data(x, y, alpha=0.2):
    """Performs Beta-distributed Mixup interpolation on input tensors and target labels."""
    if alpha <= 0.0:
        return x, y, y, 1.0
    lam = float(_MIXUP_RNG.beta(alpha, alpha))
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def _legacy_evaluate_binary_triage(probs: np.ndarray, targets: np.ndarray, threshold_spec, default_th: float = 0.15) -> dict:
    """Computes multi-point operating curves and calibrated sensitivity/specificity
    for a high-stakes binary triage or cancer screening target.
    """
    operating_points = {}
    total_pos = int(targets.sum())
    for ref_th in [0.50, 0.30, 0.20, 0.15, 0.10, 0.05, 0.02]:
        bin_p = (probs >= ref_th).astype(int)
        tp = int(np.sum((bin_p == 1) & (targets == 1)))
        fn = int(np.sum((bin_p == 0) & (targets == 1)))
        fp = int(np.sum((bin_p == 1) & (targets == 0)))
        tn = int(np.sum((bin_p == 0) & (targets == 0)))
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        operating_points[f"tau_{ref_th:.2f}"] = {
            'threshold': ref_th,
            'sensitivity': round(sens, 4),
            'specificity': round(spec, 4),
            'detected': f"{tp}/{total_pos}"
        }

    effective_th = default_th
    if threshold_spec is not None:
        if isinstance(threshold_spec, str):
            th_str = threshold_spec.lower().strip()
            if th_str in ('auto', 'youden'):
                if len(np.unique(targets)) > 1:
                    fpr_arr, tpr_arr, th_arr = roc_curve(targets, probs)
                    j_scores = tpr_arr - fpr_arr
                    best_j_idx = np.argmax(j_scores)
                    effective_th = float(th_arr[best_j_idx])
                    effective_th = max(0.01, min(0.90, effective_th))
                else:
                    effective_th = default_th
            elif th_str in ('sens90', 'sens_90', 'recall90'):
                if len(np.unique(targets)) > 1:
                    fpr_arr, tpr_arr, th_arr = roc_curve(targets, probs)
                    valid_idx = np.where(tpr_arr >= 0.90)[0]
                    effective_th = float(th_arr[valid_idx[0]]) if len(valid_idx) > 0 else 0.10
                else:
                    effective_th = 0.10
            elif th_str in ('sens95', 'sens_95', 'recall95'):
                if len(np.unique(targets)) > 1:
                    fpr_arr, tpr_arr, th_arr = roc_curve(targets, probs)
                    valid_idx = np.where(tpr_arr >= 0.95)[0]
                    effective_th = float(th_arr[valid_idx[0]]) if len(valid_idx) > 0 else 0.05
                else:
                    effective_th = 0.05
            else:
                try:
                    effective_th = float(threshold_spec)
                except ValueError:
                    effective_th = default_th
        else:
            effective_th = float(threshold_spec)

    bin_preds = (probs >= effective_th).astype(int)
    tp = int(np.sum((bin_preds == 1) & (targets == 1)))
    fn = int(np.sum((bin_preds == 0) & (targets == 1)))
    fp = int(np.sum((bin_preds == 1) & (targets == 0)))
    tn = int(np.sum((bin_preds == 0) & (targets == 0)))
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

    return {
        'threshold': round(effective_th, 4),
        'sensitivity': round(sens, 4),
        'specificity': round(spec, 4),
        'f1': round(f1, 4),
        'detected': f"{tp}/{total_pos}",
        'operating_points': operating_points
    }


def evaluate_dataset(
    model,
    loader,
    device,
    precision_dtype,
    has_cuda,
    criterion=None,
    mel_threshold=None,
    bcc_threshold='youden',
    malignant_threshold=None,
    logit_adjust=0.0,
    class_priors=None,
    use_tta=False,
    autocast=True,
    temperature=1.0,
):
    """Runs a complete evaluation pass on a dataset loader with optional Logit Adjustment,
    Triage Thresholding, and Test-Time Augmentation (TTA).
    
    Args:
        criterion: Loss function to use for validation loss computation.
        mel_threshold: Operating sensitivity threshold for melanoma triage.
        bcc_threshold: Operating sensitivity threshold for Basal Cell Carcinoma triage.
        malignant_threshold: Operating threshold for 2-tier malignancy screening (MEL + BCC + AKIEC).
        logit_adjust: Strength of post-hoc Bayesian prior correction (0.0 to 1.0).
        class_priors: Training set prior distribution array for Logit Adjustment.
        use_tta: If True, averages probabilities over original, horizontal-flip, and vertical-flip views.
        temperature: Scalar temperature applied to logits before softmax (1.0 = raw model output).
    """
    model.eval()
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a positive finite value")
    all_preds, all_targets, all_probs, all_logits = [], [], [], []
    v_loss, v_corr, v_tot = 0.0, 0, 0
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    log_priors = None
    if logit_adjust > 0.0 and class_priors is not None:
        p_tensor = torch.tensor(class_priors, dtype=torch.float32, device=device).clamp(min=1e-7)
        log_priors = torch.log(p_tensor).unsqueeze(0)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            views = [x, torch.flip(x, dims=[-1]), torch.flip(x, dims=[-2])] if use_tta else [x]
            outputs = []
            with torch.amp.autocast('cuda', dtype=precision_dtype) if has_cuda and autocast else nullcontext():
                for view in views:
                    outputs.append(model(view))
                loss = criterion(torch.stack(outputs).mean(dim=0), y)

            raw_outputs = [output.float() for output in outputs]
            float_outputs = [output / temperature for output in raw_outputs]
            probs = torch.stack([torch.softmax(output, dim=1) for output in float_outputs]).mean(dim=0)
            decision_outputs = [output - float(logit_adjust) * log_priors for output in float_outputs] if log_priors is not None else float_outputs
            decision_probs = torch.stack([torch.softmax(output, dim=1) for output in decision_outputs]).mean(dim=0)
            preds = decision_probs.argmax(dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_logits.extend(torch.stack(raw_outputs).mean(dim=0).cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            v_loss += loss.item() * len(y)
            v_corr += (preds == y).sum().item()
            v_tot += len(y)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    all_logits = np.array(all_logits)
    report = classification_report(all_targets, all_preds, labels=range(NUM_CLASSES), target_names=CLASS_NAMES, output_dict=True, zero_division=0)

    # 1. Continuous Melanoma AUC-ROC
    mel_idx = CLASS_NAMES.index('mel')
    binary_mel_targets = (all_targets == mel_idx).astype(int)
    mel_probs = all_probs[:, mel_idx]
    if len(np.unique(binary_mel_targets)) > 1:
        mel_auc_roc = float(roc_auc_score(binary_mel_targets, mel_probs))
    else:
        mel_auc_roc = 0.0

    # 2. Continuous Basal Cell Carcinoma AUC-ROC & Per-Class AUC
    per_class_auc = {}
    valid_aucs = []
    for cls_name in CLASS_NAMES:
        c_idx = CLASS_NAMES.index(cls_name)
        c_bin_targets = (all_targets == c_idx).astype(int)
        c_probs = all_probs[:, c_idx]
        if len(np.unique(c_bin_targets)) > 1:
            score = float(roc_auc_score(c_bin_targets, c_probs))
            per_class_auc[cls_name] = round(score, 4)
            valid_aucs.append(score)
        else:
            per_class_auc[cls_name] = 0.0
    macro_auc_roc = float(np.mean(valid_aucs)) if len(valid_aucs) > 0 else 0.0

    # 3. Harmonized 5-Class Granular Metrics (Shared between HAM10000 and PAD-UFES-20: mel, bcc, akiec, nv, bkl)
    shared_indices = [CLASS_NAMES.index(c) for c in ['akiec', 'bcc', 'bkl', 'mel', 'nv']]
    mask_shared = np.isin(all_targets, shared_indices)
    if mask_shared.sum() > 0:
        shared_targets = all_targets[mask_shared]
        shared_preds = all_preds[mask_shared]
        harmonized_5class_acc = float((shared_preds == shared_targets).mean())
    else:
        harmonized_5class_acc = float(report['accuracy'])

    # 4. ── Melanoma Clinical Triage Evaluation ──
    mel_res = _evaluate_binary_triage(mel_probs, binary_mel_targets, mel_threshold, default_th=0.15)

    # 5. ── Basal Cell Carcinoma (BCC) Clinical Triage Evaluation ──
    bcc_idx = CLASS_NAMES.index('bcc')
    binary_bcc_targets = (all_targets == bcc_idx).astype(int)
    bcc_probs = all_probs[:, bcc_idx]
    bcc_res = _evaluate_binary_triage(bcc_probs, binary_bcc_targets, bcc_threshold, default_th=0.15)

    # 6. ── Joint Malignancy Screening Evaluation (MEL + BCC + AKIEC) ──
    mal_indices = [CLASS_NAMES.index(c) for c in ['mel', 'bcc', 'akiec']]
    binary_mal_targets = np.isin(all_targets, mal_indices).astype(int)
    mal_probs = np.clip(all_probs[:, mal_indices].sum(axis=1), 0.0, 1.0)
    mal_res = _evaluate_binary_triage(mal_probs, binary_mal_targets, malignant_threshold, default_th=0.25)
    
    return {
        'loss': v_loss / max(v_tot, 1),
        'accuracy': float(report['accuracy']),
        'weighted_avg_f1': float(report['weighted avg']['f1-score']),
        'macro_avg_f1': float(report['macro avg']['f1-score']),
        'mel_recall': float(report.get('mel', {}).get('recall', 0.0)),
        'bcc_recall': float(report.get('bcc', {}).get('recall', 0.0)),
        'akiec_recall': float(report.get('akiec', {}).get('recall', 0.0)),
        'mel_auc_roc': mel_auc_roc,
        'macro_auc_roc': macro_auc_roc,
        'per_class_auc': per_class_auc,
        'harmonized_5class_acc': harmonized_5class_acc,
        'expected_calibration_error': expected_calibration_error(all_probs, all_targets),
        'mean_mel_probability': float(mel_probs.mean()),

        # Melanoma Triage Metrics
        'mel_triage_recall': mel_res['sensitivity'],
        'mel_triage_spec': mel_res['specificity'],
        'mel_triage_f1': mel_res['f1'],
        'mel_triage_threshold': mel_res['threshold'],
        'mel_threshold_source': mel_res.get('threshold_source'),
        'mel_threshold_fallback': mel_res.get('threshold_fallback', False),
        'mel_triage_detected': mel_res['detected'],
        'mel_operating_points': mel_res['operating_points'],

        # Basal Cell Carcinoma Triage Metrics
        'bcc_triage_recall': bcc_res['sensitivity'],
        'bcc_triage_spec': bcc_res['specificity'],
        'bcc_triage_f1': bcc_res['f1'],
        'bcc_triage_threshold': bcc_res['threshold'],
        'bcc_threshold_source': bcc_res.get('threshold_source'),
        'bcc_threshold_fallback': bcc_res.get('threshold_fallback', False),
        'bcc_triage_detected': bcc_res['detected'],
        'bcc_operating_points': bcc_res['operating_points'],

        # Joint Malignancy Screening Metrics (MEL + BCC + AKIEC)
        'malignant_triage_recall': mal_res['sensitivity'],
        'malignant_triage_spec': mal_res['specificity'],
        'malignant_triage_f1': mal_res['f1'],
        'malignant_triage_threshold': mal_res['threshold'],
        'malignant_threshold_source': mal_res.get('threshold_source'),
        'malignant_threshold_fallback': mal_res.get('threshold_fallback', False),
        'malignant_triage_detected': mal_res['detected'],
        'malignant_operating_points': mal_res['operating_points'],

        'all_preds': all_preds,
        'all_targets': all_targets,
        'all_probs': all_probs,
        'all_logits': all_logits,
        'temperature': temperature,
        'report': report
    }


def evaluation_accuracy_consistency(final_accuracy: float, selected_epoch_accuracy: float, tolerance: float = 0.05) -> tuple[float, bool]:
    delta = float(final_accuracy) - float(selected_epoch_accuracy)
    return delta, abs(delta) > tolerance


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def add_confidence_intervals(results: dict, frame: pd.DataFrame, seed: int = 42, n_bootstrap: int = 1000):
    if 'lesion_id' not in frame.columns:
        raise KeyError("Cluster-bootstrap evaluation requires lesion_id")
    groups = frame['lesion_id'].astype(str).to_numpy()
    targets = results['all_targets']
    probs = results['all_probs']
    macro_ci = bootstrap_metric_ci(
        probs, targets, groups,
        lambda p, y: float(np.mean([
            roc_auc_score((y == idx).astype(int), p[:, idx])
            for idx in range(NUM_CLASSES) if len(np.unique((y == idx).astype(int))) > 1
        ])),
        n=n_bootstrap, seed=seed
    )
    results['macro_auc_ci_low'], results['macro_auc_ci_high'] = macro_ci
    for class_name in ('mel', 'bcc'):
        class_idx = CLASS_NAMES.index(class_name)
        binary_targets = (targets == class_idx).astype(int)
        class_probs = probs[:, class_idx]
        auc_ci = bootstrap_metric_ci(
            class_probs, binary_targets, groups,
            lambda p, y: roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
            n=n_bootstrap, seed=seed
        )
        threshold = results[f'{class_name}_triage_threshold']
        sensitivity_ci = bootstrap_metric_ci(
            class_probs, binary_targets, groups,
            lambda p, y, th=threshold: float(np.sum((p >= th) & (y == 1)) / max(np.sum(y == 1), 1)),
            n=n_bootstrap, seed=seed
        )
        results[f'{class_name}_auc_ci_low'], results[f'{class_name}_auc_ci_high'] = auc_ci
        results[f'{class_name}_triage_recall_ci_low'], results[f'{class_name}_triage_recall_ci_high'] = sensitivity_ci
    return results


def save_prediction_artifacts(domain_dir: Path, results: dict, frame: pd.DataFrame):
    domain_dir.mkdir(parents=True, exist_ok=True)
    np.save(domain_dir / 'all_probs.npy', results['all_probs'])
    np.save(domain_dir / 'all_targets.npy', results['all_targets'])
    if 'all_logits' in results:
        np.save(domain_dir / 'all_logits.npy', results['all_logits'])
    if 'lesion_id' not in frame.columns:
        raise KeyError("Cluster-bootstrap evaluation requires lesion_id")
    groups = frame['lesion_id'].astype(str).to_numpy().astype(str)
    np.save(domain_dir / 'lesion_ids.npy', groups)


def compute_decision_metrics(probs, targets, class_priors, tau, malignant_threshold):
    malignant = [CLASS_NAMES.index(name) for name in ('akiec', 'bcc', 'mel')]
    decisions = {
        'argmax': decide_argmax(probs),
        'prior_corrected': decide_prior_corrected(probs, class_priors, tau),
        'malignant_gated': decide_malignant_gated(probs, malignant_threshold, malignant),
    }
    summaries = {name: confusion_summary(targets, predictions, CLASS_NAMES, malignant) for name, predictions in decisions.items()}
    metrics = {}
    for name, summary in summaries.items():
        metrics[name] = {
            'accuracy': summary['accuracy'],
            'balanced_accuracy': summary['balanced_accuracy'],
            'macro_f1': summary['macro_f1'],
            'malignant_sensitivity': summary['malignant']['sensitivity'],
            'malignant_specificity': summary['malignant']['specificity'],
        }
    return metrics, decisions, summaries


def save_decision_artifacts(domain_dir, results, class_priors, tau, malignant_threshold, model_name):
    metrics, decisions, summaries = compute_decision_metrics(
        results['all_probs'], results['all_targets'], class_priors, tau, malignant_threshold
    )
    np.save(domain_dir / 'confusion_argmax.npy', summaries['argmax']['counts'])
    np.save(domain_dir / 'confusion_tau.npy', summaries['prior_corrected']['counts'])
    np.save(domain_dir / 'confusion_gated.npy', summaries['malignant_gated']['counts'])
    plot_decision_confusion_matrices(
        results['all_probs'], results['all_targets'], CLASS_NAMES, class_priors, tau, malignant_threshold,
        domain_dir / 'confusion_matrix_decision.png', model_name=model_name
    )
    return metrics


def train_single_model(
    model_name: str,
    args,
    output_dir: Path,
    hw: dict,
    train_df: pd.DataFrame,
    ham_val_df: pd.DataFrame,
    ham_test_df: pd.DataFrame,
    pad_val_df: pd.DataFrame
) -> dict:
    """Trains a single model architecture through Stage 1 & Stage 2 and performs dual-domain validation."""
    set_seed(int(args.seed))
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    device = hw['device']
    has_cuda = (device.type == 'cuda')
    precision_dtype = hw['precision_dtype']
    use_scaler = has_cuda and not hw['has_bf16']
    scaler = torch.amp.GradScaler('cuda') if use_scaler else None

    # Adaptive Batch Strategy
    batch_strategy = compute_adaptive_batch_strategy(
        vram_gb=hw['vram_gb'], model_name=model_name, requested_batch=args.batch_size
    )
    micro_batch = batch_strategy['micro_batch']
    grad_accum_steps = batch_strategy['grad_accum_steps']

    cfg = MODEL_CONFIGS[model_name]
    img_size = args.img_size or cfg['input_size']
    timm_name = cfg['timm_name']

    lr_stage1 = args.lr_stage1 or cfg['default_lr1']
    lr_stage2 = args.lr_stage2 or cfg['default_lr2']
    weight_decay = cfg['weight_decay']

    print(f"\n{'='*80}")
    print(f" [Dual-Domain Multi-Phase Benchmark Pipeline]")
    print(f" Device: {hw['device_name']} ({hw['vram_gb']:.1f} GB VRAM, {hw['device_count']} GPU(s)) | Precision: {hw['precision_name']} | cuDNN: {'enabled' if hw['cudnn_enabled'] else 'disabled'}")
    print(f" Model: {model_name.upper()} ({timm_name}) | Pretrained: {cfg['pretrained']}")
    print(f" Batch: micro={micro_batch}, accumulation={grad_accum_steps}, effective={micro_batch * grad_accum_steps}")
    print(f" Datasets: Train (HAM: {len(train_df)}) | Val (HAM: {len(ham_val_df)}) | Test (HAM: {len(ham_test_df)}) | OOD (PAD: {len(pad_val_df)})")
    print(f"{'='*80}\n")

    model = timm.create_model(timm_name, pretrained=True, num_classes=NUM_CLASSES)
    if hw['device_count'] > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model

    is_large_model = model_name in ('v5', 'v4convl')
    if hasattr(raw_model, 'set_grad_checkpointing') and is_large_model:
        try:
            raw_model.set_grad_checkpointing(enable=True)
            print("  ⚡ Gradient Checkpointing: ENABLED (activation memory reduced by ~60%)")
        except Exception as e:
            print(f"  ⚠️ Could not enable gradient checkpointing: {e}")

    use_color_constancy = getattr(args, 'color_constancy', False)
    use_tta = getattr(args, 'use_tta', False)
    eval_precision = str(getattr(args, 'eval_precision', 'fp32')).lower()
    eval_autocast = eval_precision in ('amp', 'bf16', 'fp16')
    eval_kwargs = {
        'logit_adjust': 0.0,
        'class_priors': compute_class_priors(train_df['dx']),
        'use_tta': use_tta,
        'autocast': eval_autocast,
    }

    train_transform, preprocessing_cfg = build_transforms(
        raw_model, img_size, train=True, color_constancy=use_color_constancy
    )
    val_transform, _ = build_transforms(
        raw_model, img_size, train=False, color_constancy=use_color_constancy
    )
    print(f"  Preprocessing: size={img_size}, mean={preprocessing_cfg['mean']}, std={preprocessing_cfg['std']}, interpolation={preprocessing_cfg['interpolation']}")

    if use_color_constancy:
        print("  🌈 Illumination Constancy: ENABLED (Shades-of-Gray Minkowski p=6.0)")

    if use_tta:
        print("  🔄 Test-Time Augmentation (TTA): ENABLED (3-view probability averaging)")

    balanced_sampling = getattr(args, 'balanced_sampling', False)
    logit_adjust = float(getattr(args, 'logit_adjust', 0.0) or 0.0)
    mixup_alpha = float(getattr(args, 'mixup_alpha', 0.0) or 0.0)
    train_class_priors = compute_class_priors(train_df['dx'])

    train_ds = SkinDataset(train_df, transform=train_transform)
    ham_val_ds = SkinDataset(ham_val_df, transform=val_transform)
    ham_test_ds = SkinDataset(ham_test_df, transform=val_transform)
    pad_val_ds = SkinDataset(pad_val_df, transform=val_transform)

    num_workers = min(8, os.cpu_count() or 1)
    use_pin = (has_cuda and model_name != 'v5')
    loader_generator = torch.Generator().manual_seed(int(args.seed))
    worker_init_fn = partial(seed_worker, seed=int(args.seed))

    if balanced_sampling:
        class_counts = train_df['dx'].value_counts().to_dict()
        # Smoothed inverse weighting for balanced representation
        sample_weights = [1.0 / (max(class_counts.get(dx, 1), 1) ** 0.5) for dx in train_df['dx']]
        total_samples = len(train_df)
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=total_samples, replacement=True, generator=loader_generator
        )
        train_loader = DataLoader(
            train_ds, batch_size=micro_batch, sampler=sampler, num_workers=num_workers, pin_memory=use_pin,
            drop_last=True, worker_init_fn=worker_init_fn, persistent_workers=num_workers > 0
        )
        print(f"  ⚖️  Dynamic Augmented Sampling: ENABLED ({total_samples} samples/epoch, drop_last=True)")
    else:
        train_loader = DataLoader(
            train_ds, batch_size=micro_batch, shuffle=True, num_workers=num_workers, pin_memory=use_pin,
            drop_last=True, generator=loader_generator, worker_init_fn=worker_init_fn, persistent_workers=num_workers > 0
        )

    if logit_adjust > 0.0:
        print(f"  🎯 Logit Prior Adjustment: ENABLED (tau={logit_adjust:.2f})")
    if mixup_alpha > 0.0:
        print(f"  🎨 Mixup Augmentation: ENABLED (alpha={mixup_alpha:.2f})")

    eval_loader_kwargs = {
        'batch_size': micro_batch * 2, 'shuffle': False, 'num_workers': num_workers, 'pin_memory': use_pin,
        'worker_init_fn': worker_init_fn, 'persistent_workers': num_workers > 0
    }
    ham_val_loader = DataLoader(ham_val_ds, **eval_loader_kwargs)
    ham_test_loader = DataLoader(ham_test_ds, **eval_loader_kwargs)
    pad_val_loader = DataLoader(pad_val_ds, **eval_loader_kwargs)

    if balanced_sampling:
        # Batches are already dynamically balanced; use neutral alpha to avoid compounding penalties
        weight_tensor = torch.ones(NUM_CLASSES, dtype=torch.float32).to(device)
    else:
        weights_dict = compute_class_weights(train_df['dx'])
        weight_tensor = torch.tensor([weights_dict[i] for i in range(NUM_CLASSES)], dtype=torch.float32).to(device)
    loss_name = str(getattr(args, 'loss', 'focal') or 'focal').lower()
    if loss_name not in ('focal', 'ce'):
        raise ValueError(f"Unsupported loss '{loss_name}' (expected 'focal' or 'ce')")
    # gamma=0 reduces focal loss to alpha-weighted cross-entropy, so both losses share the same weighting rule.
    focal_gamma = 2.0 if loss_name == 'focal' else 0.0
    criterion = PyTorchFocalLoss(alpha=weight_tensor, gamma=focal_gamma)
    print(f"  Loss: {'Focal (gamma=2.0)' if loss_name == 'focal' else 'Cross-entropy (focal gamma=0)'} | alpha: {'neutral' if balanced_sampling else 'inverse-frequency'}")

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 1: Warmup Classifier Head (Backbone Frozen)
    # ──────────────────────────────────────────────────────────────────────────
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    for param in raw_model.parameters():
        param.requires_grad = False
    head = raw_model.get_classifier()
    for param in head.parameters():
        param.requires_grad = True

    stage1_epochs = max(int(getattr(args, 'stage1_epochs', 3)), 1)
    optimizer1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_stage1, weight_decay=weight_decay)
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min', factor=0.3, patience=2, min_lr=1e-7)

    print(f"\n--- Stage 1: Linear Warmup (Head Only, lr={lr_stage1}, epochs={stage1_epochs}) ---")
    h1 = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': [], 'time_per_epoch': []}
    best_s1_loss = float('inf')

    for epoch in range(stage1_epochs):
        t0 = perf_counter()
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        optimizer1.zero_grad()

        pbar = tqdm(train_loader, desc=f"Stage 1 Epoch {epoch+1}/{stage1_epochs}", leave=False)
        for step, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            if mixup_alpha > 0.0 and model.training:
                inputs_m, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=mixup_alpha)
                if has_cuda:
                    with torch.amp.autocast('cuda', dtype=precision_dtype):
                        outputs = model(inputs_m)
                        loss = (lam * criterion(outputs, targets_a) + (1.0 - lam) * criterion(outputs, targets_b)) / grad_accum_steps
                    if use_scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                else:
                    outputs = model(inputs_m)
                    loss = (lam * criterion(outputs, targets_a) + (1.0 - lam) * criterion(outputs, targets_b)) / grad_accum_steps
                    loss.backward()
            else:
                if has_cuda:
                    with torch.amp.autocast('cuda', dtype=precision_dtype):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets) / grad_accum_steps
                    if use_scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets) / grad_accum_steps
                    loss.backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                if use_scaler:
                    scaler.step(optimizer1)
                    scaler.update()
                else:
                    optimizer1.step()
                optimizer1.zero_grad()

            total_loss += loss.item() * grad_accum_steps * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += inputs.size(0)
            if (step + 1) % 10 == 0 or (step + 1) == len(train_loader):
                pbar.set_postfix({'loss': f"{total_loss / total:.4f}", 'acc': f"{correct / total:.4f}"})

        epoch_time = perf_counter() - t0
        train_acc = correct / total
        train_loss = total_loss / total

        val_eval = evaluate_dataset(
            model, ham_val_loader, device, precision_dtype, has_cuda, criterion=criterion,
            **eval_kwargs
        )
        val_acc, val_loss = val_eval['accuracy'], val_eval['loss']
        scheduler1.step(val_loss)

        h1['accuracy'].append(train_acc)
        h1['val_accuracy'].append(val_acc)
        h1['loss'].append(train_loss)
        h1['val_loss'].append(val_loss)
        h1['time_per_epoch'].append(epoch_time)

        print(f"  Stage 1 Epoch {epoch+1:02d}/{stage1_epochs:02d} [{epoch_time:.1f}s] "
              f"— Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
              f"| lr: {optimizer1.param_groups[0]['lr']:.2e}")

        if val_loss < best_s1_loss:
            best_s1_loss = val_loss
            torch.save(raw_model.state_dict(), model_dir / 'stage1_head_best.pth')

    stage1_checkpoint = model_dir / 'stage1_head_best.pth'
    if stage1_checkpoint.exists():
        raw_model.load_state_dict(torch.load(stage1_checkpoint, map_location=device, weights_only=True))

    s1_ham_eval = evaluate_dataset(
        model, ham_val_loader, device, precision_dtype, has_cuda,
        **eval_kwargs
    )
    print(f"\n[Stage 1 Complete] HAM validation Acc: {s1_ham_eval['accuracy']:.2%}\n")
    del optimizer1
    if has_cuda:
        torch.cuda.empty_cache()

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 2: End-to-End Fine-Tuning
    # ──────────────────────────────────────────────────────────────────────────
    if model_name == 'v5':
        for param in raw_model.parameters():
            param.requires_grad = False
        if len(raw_model.blocks) < 4:
            raise RuntimeError(f"Unexpected MobileNetV5 stage count: {len(raw_model.blocks)}")
        modules_to_unfreeze = [raw_model.blocks[2], raw_model.blocks[3], raw_model.msfa, raw_model.get_classifier()]
        for module in modules_to_unfreeze:
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = True
    else:
        for param in raw_model.parameters():
            param.requires_grad = True
    trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in raw_model.parameters())
    assert trainable_params > 0, f"No trainable parameters selected for {model_name}"
    print(f" Trainable parameters: {trainable_params:,}/{total_params:,}")

    optimizer2 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_stage2, weight_decay=weight_decay)
    optimizer_trainable_params = sum(p.numel() for group in optimizer2.param_groups for p in group['params'])
    assert optimizer_trainable_params == trainable_params, "Optimizer parameter count does not match trainable parameter count"
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='max', factor=0.3, patience=3, min_lr=1e-8)

    print(f"--- Stage 2: Backbone Fine-Tuning (lr={lr_stage2}, epochs={args.epochs}, patience={args.patience}) ---")
    h2 = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': [], 'val_mel_auc': [], 'val_macro_auc': [], 'lr': [], 'time_per_epoch': []}
    best_val_loss = float('inf')
    best_val_auc = float('-inf')
    best_selection_score = float('-inf')
    selection_min_delta = float(getattr(args, 'selection_min_delta', 0.0005))
    selected_epoch = 0
    patience_counter = 0
    checkpoint_path = model_dir / 'best_model.pth'

    for epoch in range(args.epochs):
        t0 = perf_counter()
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        optimizer2.zero_grad()

        pbar = tqdm(train_loader, desc=f"Stage 2 Epoch {epoch+1}/{args.epochs}", leave=False)
        for step, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            if mixup_alpha > 0.0 and model.training:
                inputs_m, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=mixup_alpha)
                if has_cuda:
                    with torch.amp.autocast('cuda', dtype=precision_dtype):
                        outputs = model(inputs_m)
                        loss = (lam * criterion(outputs, targets_a) + (1.0 - lam) * criterion(outputs, targets_b)) / grad_accum_steps
                    if use_scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                else:
                    outputs = model(inputs_m)
                    loss = (lam * criterion(outputs, targets_a) + (1.0 - lam) * criterion(outputs, targets_b)) / grad_accum_steps
                    loss.backward()
            else:
                if has_cuda:
                    with torch.amp.autocast('cuda', dtype=precision_dtype):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets) / grad_accum_steps
                    if use_scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets) / grad_accum_steps
                    loss.backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                if use_scaler:
                    scaler.step(optimizer2)
                    scaler.update()
                else:
                    optimizer2.step()
                optimizer2.zero_grad()

            total_loss += loss.item() * grad_accum_steps * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += inputs.size(0)
            if (step + 1) % 10 == 0 or (step + 1) == len(train_loader):
                pbar.set_postfix({'loss': f"{total_loss / total:.4f}", 'acc': f"{correct / total:.4f}"})

        epoch_time = perf_counter() - t0
        train_acc = correct / total
        train_loss = total_loss / total

        val_eval = evaluate_dataset(
            model, ham_val_loader, device, precision_dtype, has_cuda, criterion=criterion,
            **eval_kwargs
        )
        val_acc, val_loss = val_eval['accuracy'], val_eval['loss']
        selection_score = 0.5 * val_eval['mel_auc_roc'] + 0.5 * val_eval['macro_auc_roc']
        scheduler2.step(selection_score)

        h2['accuracy'].append(train_acc)
        h2['val_accuracy'].append(val_acc)
        h2['loss'].append(train_loss)
        h2['val_loss'].append(val_loss)
        h2['val_mel_auc'].append(val_eval['mel_auc_roc'])
        h2['val_macro_auc'].append(val_eval['macro_auc_roc'])
        h2['lr'].append(optimizer2.param_groups[0]['lr'])
        h2['time_per_epoch'].append(epoch_time)

        print(f"  Stage 2 Epoch {epoch+1:02d}/{args.epochs:02d} [{epoch_time:.1f}s] "
              f"— Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
              f"| lr: {optimizer2.param_groups[0]['lr']:.2e}")

        val_auc = val_eval['mel_auc_roc']
        improved = selection_score > best_selection_score + selection_min_delta or (np.isclose(selection_score, best_selection_score) and val_loss < best_val_loss)
        if improved:
            best_val_auc = val_auc
            best_selection_score = selection_score
            best_val_loss = val_loss
            selected_epoch = epoch + 1
            patience_counter = 0
            torch.save(raw_model.state_dict(), checkpoint_path)
            print(f"  ⭐ New best checkpoint saved (selection: {selection_score:.4f}, Mel AUC: {val_auc:.4f}, val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[EarlyStopping] Validation composite AUC did not improve for {args.patience} epochs. Stopping.")
                break

    if checkpoint_path.exists():
        raw_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    print(f"\n{'='*80}")
    print(f" 🏆 Running Final Dual-Domain Benchmark Evaluation on {model_name.upper()}")
    print(f"{'='*80}\n")

    mel_th = getattr(args, 'mel_threshold', None)
    bcc_th = getattr(args, 'bcc_threshold', 'youden')
    mal_th = getattr(args, 'malignant_threshold', None)

    # Pass 1 (T=1): raw HAM-val outputs. Used for the loop-consistency guard and to fit the temperature.
    ham_val_raw = evaluate_dataset(
        model, ham_val_loader, device, precision_dtype, has_cuda,
        mel_threshold=mel_th, bcc_threshold=bcc_th, malignant_threshold=mal_th,
        **eval_kwargs
    )
    use_temperature_scaling = bool(getattr(args, 'temperature_scaling', True))
    temperature = fit_temperature(ham_val_raw['all_logits'], ham_val_raw['all_targets']) if use_temperature_scaling else 1.0
    ham_val_ece_before_ts = ham_val_raw['expected_calibration_error']
    eval_kwargs['temperature'] = temperature
    print(f"  Temperature scaling: {'T=%.3f (fit on HAM-val NLL)' % temperature if use_temperature_scaling else 'disabled'}")

    # Pass 2 (T fitted): every threshold, tau and downstream metric is calibrated on temperature-scaled probabilities.
    ham_val_results = evaluate_dataset(
        model, ham_val_loader, device, precision_dtype, has_cuda,
        mel_threshold=mel_th, bcc_threshold=bcc_th, malignant_threshold=mal_th,
        **eval_kwargs
    ) if use_temperature_scaling else ham_val_raw
    selected_loop_accuracy = h2['val_accuracy'][selected_epoch - 1]
    ham_val_accuracy_delta, eval_consistency_warning = evaluation_accuracy_consistency(
        ham_val_results['accuracy'], selected_loop_accuracy
    )
    if eval_consistency_warning:
        print(f"  WARNING: final HAM validation accuracy differs from selected-epoch accuracy by {ham_val_accuracy_delta:+.2%}")
    ham_val_results = add_confidence_intervals(ham_val_results, ham_val_df, seed=int(args.seed))
    calibrated_th = ham_val_results['mel_triage_threshold']
    calibrated_bcc_th = ham_val_results['bcc_triage_threshold']
    calibrated_mal_th = ham_val_results['malignant_triage_threshold']
    class_priors = np.asarray(eval_kwargs['class_priors'], dtype=float)
    auto_tau, tau_sweep = select_logit_adjust(ham_val_results['all_probs'], ham_val_results['all_targets'], class_priors)
    selected_tau = float(logit_adjust) if logit_adjust > 0.0 else auto_tau
    tau_source = 'manual' if logit_adjust > 0.0 else 'ham_val_balanced_accuracy'
    ham_val_dir = model_dir / 'ham_validation'
    save_prediction_artifacts(ham_val_dir, ham_val_results, ham_val_df)
    ham_val_decision_metrics, _, _ = compute_decision_metrics(
        ham_val_results['all_probs'], ham_val_results['all_targets'], class_priors, selected_tau, calibrated_mal_th
    )

    ham_results = evaluate_dataset(
        model, ham_test_loader, device, precision_dtype, has_cuda,
        mel_threshold=calibrated_th, bcc_threshold=calibrated_bcc_th, malignant_threshold=calibrated_mal_th,
        **eval_kwargs
    )
    ham_results = add_confidence_intervals(ham_results, ham_test_df, seed=int(args.seed))
    ham_dir = model_dir / 'ham10000'
    save_prediction_artifacts(ham_dir, ham_results, ham_test_df)
    if use_tta:
        ham_no_tta = evaluate_dataset(
            model, ham_test_loader, device, precision_dtype, has_cuda,
            mel_threshold=calibrated_th, bcc_threshold=calibrated_bcc_th, malignant_threshold=calibrated_mal_th,
            **{**eval_kwargs, 'use_tta': False}
        )
        np.save(ham_dir / 'all_probs_no_tta.npy', ham_no_tta['all_probs'])
    with open(ham_dir / 'classification_report.json', 'w') as f:
        json.dump(ham_results['report'], f, indent=2)
    plot_confusion_matrices(ham_results['all_targets'], ham_results['all_preds'], CLASS_NAMES, ham_dir / 'confusion_matrix.png', model_name=f"{model_name} (HAM10000 Test)")
    plot_per_class_metrics(ham_results['report'], CLASS_NAMES, ham_dir / 'per_class_metrics.png', model_name=f"{model_name} (HAM10000 Test)")
    plot_roc_curves(ham_results['all_targets'], ham_results['all_probs'], CLASS_NAMES, ham_dir / 'roc_curves.png', model_name=f"{model_name} (HAM10000 Test)")
    plot_reliability_diagram(ham_results['all_probs'], ham_results['all_targets'], ham_dir / 'reliability_diagram.png', model_name=f"{model_name} (HAM10000 Test)")
    ham_decision_metrics = save_decision_artifacts(
        ham_dir, ham_results, class_priors, selected_tau, calibrated_mal_th, f"{model_name} (HAM10000 Test)"
    )
    generate_gradcam_gallery(model=model, val_df=ham_test_df, class_names=CLASS_NAMES, img_size=img_size, output_path=ham_dir / 'gradcam_heatmaps.png', model_name=f"{model_name}_HAM", device=device, transform=val_transform)

    pad_results = evaluate_dataset(
        model, pad_val_loader, device, precision_dtype, has_cuda,
        mel_threshold=calibrated_th, bcc_threshold=calibrated_bcc_th, malignant_threshold=calibrated_mal_th,
        **eval_kwargs
    )
    pad_results = add_confidence_intervals(pad_results, pad_val_df, seed=int(args.seed))
    pad_dir = model_dir / 'pad_ufes_20'
    save_prediction_artifacts(pad_dir, pad_results, pad_val_df)
    if use_tta:
        pad_no_tta = evaluate_dataset(
            model, pad_val_loader, device, precision_dtype, has_cuda,
            mel_threshold=calibrated_th, bcc_threshold=calibrated_bcc_th, malignant_threshold=calibrated_mal_th,
            **{**eval_kwargs, 'use_tta': False}
        )
        np.save(pad_dir / 'all_probs_no_tta.npy', pad_no_tta['all_probs'])
    with open(pad_dir / 'classification_report.json', 'w') as f:
        json.dump(pad_results['report'], f, indent=2)
    plot_confusion_matrices(pad_results['all_targets'], pad_results['all_preds'], CLASS_NAMES, pad_dir / 'confusion_matrix.png', model_name=f"{model_name} (PAD-UFES-20)")
    plot_per_class_metrics(pad_results['report'], CLASS_NAMES, pad_dir / 'per_class_metrics.png', model_name=f"{model_name} (PAD-UFES-20)")
    plot_roc_curves(pad_results['all_targets'], pad_results['all_probs'], CLASS_NAMES, pad_dir / 'roc_curves.png', model_name=f"{model_name} (PAD-UFES-20)")
    plot_reliability_diagram(pad_results['all_probs'], pad_results['all_targets'], pad_dir / 'reliability_diagram.png', model_name=f"{model_name} (PAD-UFES-20)")
    pad_decision_metrics = save_decision_artifacts(
        pad_dir, pad_results, class_priors, selected_tau, calibrated_mal_th, f"{model_name} (PAD-UFES-20)"
    )
    generate_gradcam_gallery(model=model, val_df=pad_val_df, class_names=CLASS_NAMES, img_size=img_size, output_path=pad_dir / 'gradcam_heatmaps.png', model_name=f"{model_name}_PAD", device=device, transform=val_transform)

    shared_indices = [CLASS_NAMES.index(c) for c in ['akiec', 'bcc', 'bkl', 'mel', 'nv']]
    pad_results['restricted_5class_acc'] = restricted_class_accuracy(pad_results['all_probs'], pad_results['all_targets'], shared_indices)
    pad_oracle = evaluate_dataset(
        model, pad_val_loader, device, precision_dtype, has_cuda,
        mel_threshold='youden', bcc_threshold='youden', malignant_threshold='youden',
        **eval_kwargs
    )

    plot_training_curves([h1, h2], ['Stage 1 (Head)', 'Stage 2 (Fine-Tune)'], model_dir / 'training_curves.png', model_name=model_name)
    plot_domain_comparison(ham_results, pad_results, model_name=model_name, output_path=model_dir / 'domain_comparison.png')
    plot_dual_roc_comparison(ham_results, pad_results, CLASS_NAMES, model_dir / 'roc_curves_dual_domain.png', model_name=model_name)

    domain_gap = ham_results['accuracy'] - pad_results['accuracy']
    mel_auc_gap = ham_results['mel_auc_roc'] - pad_results['mel_auc_roc']
    ham_macro_auc_5class = float(np.mean([ham_results['per_class_auc'][c] for c in ['akiec', 'bcc', 'bkl', 'mel', 'nv']]))
    macro_auc_gap = ham_macro_auc_5class - pad_results['macro_auc_roc']
    mel_sens_gap = ham_results['mel_triage_recall'] - pad_results['mel_triage_recall']
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    full_results = {
        'model': model_name,
        'pretrained': cfg['pretrained'],
        'img_size': img_size,
        'batch_size': args.batch_size,
        'batch_strategy': batch_strategy,
        'params': sum(p.numel() for p in model.parameters()),
        'trainable_params': trainable_params,
        'optimizer_trainable_params': optimizer_trainable_params,
        'checkpoint_sha256': checkpoint_sha256,
        'preprocessing': preprocessing_cfg,
        'eval_precision': eval_precision,
        'selection_metric': '0.5_ham_val_mel_auc_roc_plus_0.5_ham_val_macro_auc_roc',
        'selection_score': best_selection_score,
        'selected_epoch': selected_epoch,
        'selected_epoch_loop_accuracy': selected_loop_accuracy,
        'ham_val_accuracy_delta': ham_val_accuracy_delta,
        'eval_consistency_warning': eval_consistency_warning,
        'mel_triage_threshold': calibrated_th,
        'bcc_triage_threshold': calibrated_bcc_th,
        'malignant_triage_threshold': calibrated_mal_th,
        'balanced_sampling': balanced_sampling,
        'loss': loss_name,
        'temperature_scaling': use_temperature_scaling,
        'temperature': float(temperature),
        'temperature_source': 'ham_val_nll' if use_temperature_scaling else 'disabled',
        'ham_val_ece_before_ts': float(ham_val_ece_before_ts),
        'ham_val_ece_after_ts': float(ham_val_results['expected_calibration_error']),
        'logit_adjust': logit_adjust,
        'selected_logit_adjust': selected_tau,
        'logit_adjust_source': tau_source,
        'logit_adjust_sweep': tau_sweep,
        'mixup_alpha': mixup_alpha,
        'color_constancy': use_color_constancy,
        'use_tta': use_tta,

        # Stage 1 Benchmarks
        'stage1_ham_val_accuracy': s1_ham_eval['accuracy'],

        # HAM validation selection/calibration metrics
        'ham_val_accuracy': ham_val_results['accuracy'],
        'ham_val_mel_auc_roc': ham_val_results['mel_auc_roc'],
        'ham_val_mel_auc_ci_low': ham_val_results['mel_auc_ci_low'],
        'ham_val_mel_auc_ci_high': ham_val_results['mel_auc_ci_high'],
        'ham_val_macro_auc_roc': ham_val_results['macro_auc_roc'],
        'ham_val_macro_auc_ci_low': ham_val_results['macro_auc_ci_low'],
        'ham_val_macro_auc_ci_high': ham_val_results['macro_auc_ci_high'],
        'ham_val_expected_calibration_error': ham_val_results['expected_calibration_error'],
        'ham_val_mean_mel_probability': ham_val_results['mean_mel_probability'],
        'ham_val_balanced_accuracy': ham_val_decision_metrics['argmax']['balanced_accuracy'],
        'ham_val_balanced_accuracy_tau': ham_val_decision_metrics['prior_corrected']['balanced_accuracy'],
        'ham_val_macro_f1_tau': ham_val_decision_metrics['prior_corrected']['macro_f1'],

        # Stage 2 In-Domain HAM10000 Test Metrics
        'ham_accuracy': ham_results['accuracy'],
        'ham_weighted_f1': ham_results['weighted_avg_f1'],
        'ham_macro_f1': ham_results['macro_avg_f1'],
        'ham_mel_recall': ham_results['mel_recall'],
        'ham_bcc_recall': ham_results['bcc_recall'],
        'ham_akiec_recall': ham_results['akiec_recall'],
        'ham_mel_auc_roc': ham_results['mel_auc_roc'],
        'ham_macro_auc_roc': ham_results['macro_auc_roc'],
        'ham_test_accuracy': ham_results['accuracy'],
        'ham_test_mel_auc_roc': ham_results['mel_auc_roc'],
        'ham_test_mel_auc_ci_low': ham_results['mel_auc_ci_low'],
        'ham_test_mel_auc_ci_high': ham_results['mel_auc_ci_high'],
        'ham_test_bcc_auc_roc': ham_results['per_class_auc']['bcc'],
        'ham_test_bcc_auc_ci_low': ham_results['bcc_auc_ci_low'],
        'ham_test_bcc_auc_ci_high': ham_results['bcc_auc_ci_high'],
        'ham_test_macro_auc_ci_low': ham_results['macro_auc_ci_low'],
        'ham_test_macro_auc_ci_high': ham_results['macro_auc_ci_high'],
        'ham_harmonized_5class_acc': ham_results['harmonized_5class_acc'],
        'ham_expected_calibration_error': ham_results['expected_calibration_error'],
        'ham_balanced_accuracy': ham_decision_metrics['argmax']['balanced_accuracy'],
        'ham_balanced_accuracy_tau': ham_decision_metrics['prior_corrected']['balanced_accuracy'],
        'ham_macro_f1_tau': ham_decision_metrics['prior_corrected']['macro_f1'],
        'ham_malignant_gated_sensitivity': ham_decision_metrics['malignant_gated']['malignant_sensitivity'],
        'ham_malignant_gated_specificity': ham_decision_metrics['malignant_gated']['malignant_specificity'],
        'ham_test_macro_auc_5class': ham_macro_auc_5class,
        'ham_mel_triage_recall': ham_results['mel_triage_recall'],
        'ham_mel_triage_spec': ham_results['mel_triage_spec'],
        'ham_mel_triage_recall_ci_low': ham_results['mel_triage_recall_ci_low'],
        'ham_mel_triage_recall_ci_high': ham_results['mel_triage_recall_ci_high'],
        'ham_mel_operating_points': ham_results['mel_operating_points'],
        'ham_bcc_triage_recall': ham_results['bcc_triage_recall'],
        'ham_bcc_triage_spec': ham_results['bcc_triage_spec'],
        'ham_bcc_triage_recall_ci_low': ham_results['bcc_triage_recall_ci_low'],
        'ham_bcc_triage_recall_ci_high': ham_results['bcc_triage_recall_ci_high'],
        'ham_bcc_operating_points': ham_results['bcc_operating_points'],
        'ham_malignant_triage_recall': ham_results['malignant_triage_recall'],
        'ham_malignant_triage_spec': ham_results['malignant_triage_spec'],

        # Stage 2 Out-of-Domain PAD-UFES-20 Metrics
        'pad_accuracy': pad_results['accuracy'],
        'pad_weighted_f1': pad_results['weighted_avg_f1'],
        'pad_macro_f1': pad_results['macro_avg_f1'],
        'pad_mel_recall': pad_results['mel_recall'],
        'pad_bcc_recall': pad_results['bcc_recall'],
        'pad_akiec_recall': pad_results['akiec_recall'],
        'pad_mel_auc_roc': pad_results['mel_auc_roc'],
        'pad_mel_auc_ci_low': pad_results['mel_auc_ci_low'],
        'pad_mel_auc_ci_high': pad_results['mel_auc_ci_high'],
        'pad_bcc_auc_roc': pad_results['per_class_auc']['bcc'],
        'pad_bcc_auc_ci_low': pad_results['bcc_auc_ci_low'],
        'pad_bcc_auc_ci_high': pad_results['bcc_auc_ci_high'],
        'pad_macro_auc_ci_low': pad_results['macro_auc_ci_low'],
        'pad_macro_auc_ci_high': pad_results['macro_auc_ci_high'],
        'pad_macro_auc_roc': pad_results['macro_auc_roc'],
        'pad_harmonized_5class_acc': pad_results['harmonized_5class_acc'],
        'pad_restricted_5class_acc': pad_results['restricted_5class_acc'],
        'pad_expected_calibration_error': pad_results['expected_calibration_error'],
        'pad_balanced_accuracy': pad_decision_metrics['argmax']['balanced_accuracy'],
        'pad_balanced_accuracy_tau': pad_decision_metrics['prior_corrected']['balanced_accuracy'],
        'pad_macro_f1_tau': pad_decision_metrics['prior_corrected']['macro_f1'],
        'pad_malignant_gated_sensitivity': pad_decision_metrics['malignant_gated']['malignant_sensitivity'],
        'pad_malignant_gated_specificity': pad_decision_metrics['malignant_gated']['malignant_specificity'],
        'pad_oracle_mel_triage_threshold': pad_oracle['mel_triage_threshold'],
        'pad_oracle_mel_triage_recall': pad_oracle['mel_triage_recall'],
        'pad_oracle_mel_triage_spec': pad_oracle['mel_triage_spec'],
        'pad_mel_triage_recall': pad_results['mel_triage_recall'],
        'pad_mel_triage_spec': pad_results['mel_triage_spec'],
        'pad_mel_triage_recall_ci_low': pad_results['mel_triage_recall_ci_low'],
        'pad_mel_triage_recall_ci_high': pad_results['mel_triage_recall_ci_high'],
        'pad_mel_triage_detected': pad_results['mel_triage_detected'],
        'pad_mel_operating_points': pad_results['mel_operating_points'],
        'pad_bcc_triage_recall': pad_results['bcc_triage_recall'],
        'pad_bcc_triage_spec': pad_results['bcc_triage_spec'],
        'pad_bcc_triage_recall_ci_low': pad_results['bcc_triage_recall_ci_low'],
        'pad_bcc_triage_recall_ci_high': pad_results['bcc_triage_recall_ci_high'],
        'pad_bcc_triage_detected': pad_results['bcc_triage_detected'],
        'pad_bcc_operating_points': pad_results['bcc_operating_points'],
        'pad_malignant_triage_recall': pad_results['malignant_triage_recall'],
        'pad_malignant_triage_spec': pad_results['malignant_triage_spec'],
        'pad_malignant_triage_detected': pad_results['malignant_triage_detected'],
        'pad_malignant_operating_points': pad_results['malignant_operating_points'],

        # Clinical Domain Shift Drop (In-Domain - Out-of-Domain)
        'domain_gap': float(domain_gap),
        'domain_gap_drop': float(domain_gap),
        'mel_auc_gap': float(mel_auc_gap),
        'macro_auc_gap': float(macro_auc_gap),
        'mel_sens_gap': float(mel_sens_gap),
        'mel_threshold_source': ham_val_results.get('mel_threshold_source'),
        'bcc_threshold_source': ham_val_results.get('bcc_threshold_source'),
        'malignant_threshold_source': ham_val_results.get('malignant_threshold_source'),
        'meets_auc_target_ham': meets_auc_target(ham_results['mel_auc_ci_low']),
        'meets_auc_target_pad': meets_auc_target(pad_results['mel_auc_ci_low']),
        'peak_vram_allocated_gb': torch.cuda.max_memory_allocated(device) / (1024 ** 3) if has_cuda else 0.0,
        'peak_vram_reserved_gb': torch.cuda.max_memory_reserved(device) / (1024 ** 3) if has_cuda else 0.0,
    }

    full_results = json_safe(full_results)
    with open(model_dir / 'history.json', 'w') as f:
        json.dump(json_safe({'stage1': h1, 'stage2': h2}), f, indent=2, allow_nan=False)
    with open(model_dir / 'results.json', 'w') as f:
        json.dump(full_results, f, indent=2, allow_nan=False)

    print(f"\n===========================================================================")
    print(f" 📊 Final Results Summary for {model_name.upper()}:")
    print(f"   In-Domain (HAM10000):     Acc={ham_results['accuracy']:.2%} | Macro AUC={ham_results['macro_auc_roc']:.4f} | Mel AUC={ham_results['mel_auc_roc']:.4f}")
    print(f"     ↳ Argmax:               Mel Recall={ham_results['mel_recall']:.2%} | BCC Recall={ham_results['bcc_recall']:.2%}")
    print(f"     ↳ MEL Triage (tau={calibrated_th:.2f}): Mel Recall={ham_results['mel_triage_recall']:.2%} | Spec={ham_results['mel_triage_spec']:.2%}")
    print(f"     ↳ BCC Triage (tau={calibrated_bcc_th:.2f}): BCC Recall={ham_results['bcc_triage_recall']:.2%} | Spec={ham_results['bcc_triage_spec']:.2%}")
    print(f"     ↳ Malignancy Screen:    Recall={ham_results['malignant_triage_recall']:.2%} | Spec={ham_results['malignant_triage_spec']:.2%}")
    print(f"     ↳ Calibration:          T={temperature:.3f} | HAM-val ECE {ham_val_ece_before_ts:.4f} -> {ham_val_results['expected_calibration_error']:.4f} | tau*={selected_tau:.1f} ({tau_source}) | BAcc argmax={ham_decision_metrics['argmax']['balanced_accuracy']:.2%} tau={ham_decision_metrics['prior_corrected']['balanced_accuracy']:.2%}")
    print(f"   Out-of-Domain (PAD-UFES): Acc={pad_results['accuracy']:.2%} | Macro AUC={pad_results['macro_auc_roc']:.4f} | Mel AUC={pad_results['mel_auc_roc']:.4f}")
    print(f"     ↳ Argmax:               Mel Recall={pad_results['mel_recall']:.2%} | BCC Recall={pad_results['bcc_recall']:.2%}")
    print(f"     ↳ MEL Triage (tau={calibrated_th:.2f}): Mel Recall={pad_results['mel_triage_recall']:.2%} | Spec={pad_results['mel_triage_spec']:.2%} | Detected={pad_results['mel_triage_detected']}")
    print(f"     ↳ BCC Triage (tau={calibrated_bcc_th:.2f}): BCC Recall={pad_results['bcc_triage_recall']:.2%} | Spec={pad_results['bcc_triage_spec']:.2%} | Detected={pad_results['bcc_triage_detected']}")
    print(f"     ↳ Malignancy Screen:    Recall={pad_results['malignant_triage_recall']:.2%} | Spec={pad_results['malignant_triage_spec']:.2%} | Detected={pad_results['malignant_triage_detected']}")
    print(f"   Clinical Domain Gap (Δ Acc Drop): -{domain_gap*100:.2f}%")
    print(f"   Results saved to: {model_dir / 'results.json'}")
    print(f"   Domain Comparison Chart: {model_dir / 'domain_comparison.png'}")
    print(f"   Dual-Domain ROC Analysis: {model_dir / 'roc_curves_dual_domain.png'}")
    print(f"===========================================================================\n")

    return full_results


if __name__ == '__main__':
    from main import main
    main()
