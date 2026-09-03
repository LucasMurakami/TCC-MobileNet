"""
PyTorch & timm Pretrained Models Trainer (MobileNet V1 - V5)
Dual-Domain Multi-Phase Evaluation Pipeline:
  - In-Domain Evaluation: HAM10000 (Dermoscopy)
  - Out-of-Domain Evaluation: PAD-UFES-20 (Clinical Smartphone)
  - Full tracking across Stage 1 (Warmup) and Stage 2 (Fine-Tuning)
"""

import argparse
import os
import sys
import json
import time
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
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from dataset import CLASS_NAMES, NUM_CLASSES
from visualize import (
    plot_training_curves, plot_confusion_matrices,
    plot_per_class_metrics, generate_gradcam_gallery, plot_domain_comparison,
    plot_roc_curves, plot_dual_roc_comparison
)


# ─── Modular Hardware Engine ────────────────────────────────────────────────

def configure_hardware_environment() -> dict:
    """Detects GPU architecture, compute capability, VRAM size, native BF16/FP16 support."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        device_count = torch.cuda.device_count()
        has_bf16 = torch.cuda.is_bf16_supported()
        precision_dtype = torch.bfloat16 if has_bf16 else torch.float16
        precision_name = 'BFloat16 (BF16)' if has_bf16 else 'Float16 (FP16)'
        try:
            test_x = torch.randn(1, 1, 4, 4, device=device)
            test_conv = nn.Conv2d(1, 1, 2).to(device)
            _ = test_conv(test_x)
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
    }


def compute_adaptive_batch_strategy(vram_gb: float, model_name: str, requested_batch: int = 32) -> tuple:
    """Computes physical micro-batch size and gradient accumulation steps."""
    is_large_model = model_name in ('v5', 'v4convl')

    if vram_gb >= 24:
        micro_batch = requested_batch
    elif vram_gb >= 12:
        micro_batch = 16 if is_large_model else min(requested_batch, 32)
    elif vram_gb >= 8:
        micro_batch = 8 if is_large_model else min(requested_batch, 16)
    elif vram_gb >= 4:
        micro_batch = 4 if is_large_model else min(requested_batch, 8)
    else:
        micro_batch = 2 if is_large_model else min(requested_batch, 4)

    grad_accum_steps = max(1, requested_batch // micro_batch)
    return micro_batch, grad_accum_steps




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
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
        except Exception as e:
            print(f"  [WARNING] Corrupt/missing image at index {idx}: {self.paths[idx]} — {e}. Using blank tensor.")
            # Return a blank (black) image tensor with the correct label
            if self.transform:
                img = Image.new('RGB', (256, 256), (0, 0, 0))
            else:
                return torch.zeros(3, 256, 256), label
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
    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def _evaluate_binary_triage(probs: np.ndarray, targets: np.ndarray, threshold_spec, default_th: float = 0.15) -> dict:
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
    use_tta=False
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
        use_tta: If True, computes 4-view Test-Time Augmentation (orig, hflip, vflip, rot90).
    """
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
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
            if use_tta:
                x_h = torch.flip(x, dims=[-1])
                x_v = torch.flip(x, dims=[-2])
                x_r = torch.rot90(x, 1, [-2, -1])
                with torch.amp.autocast('cuda', dtype=precision_dtype) if has_cuda else torch.nullcontext():
                    out1 = model(x)
                    out2 = model(x_h)
                    out3 = model(x_v)
                    out4 = model(x_r)
                    out = (out1 + out2 + out3 + out4) / 4.0
                    loss = criterion(out, y)
            else:
                with torch.amp.autocast('cuda', dtype=precision_dtype) if has_cuda else torch.nullcontext():
                    out = model(x)
                    loss = criterion(out, y)

            # Apply Post-Hoc Logit Adjustment (Solution B)
            if log_priors is not None:
                adj_out = out.float() - float(logit_adjust) * log_priors
            else:
                adj_out = out.float()

            probs = torch.softmax(adj_out, dim=1)
            preds = adj_out.argmax(dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            v_loss += loss.item() * len(y)
            v_corr += (preds == y).sum().item()
            v_tot += len(y)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
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

        # Melanoma Triage Metrics
        'mel_triage_recall': mel_res['sensitivity'],
        'mel_triage_spec': mel_res['specificity'],
        'mel_triage_f1': mel_res['f1'],
        'mel_triage_threshold': mel_res['threshold'],
        'mel_triage_detected': mel_res['detected'],
        'mel_operating_points': mel_res['operating_points'],

        # Basal Cell Carcinoma Triage Metrics
        'bcc_triage_recall': bcc_res['sensitivity'],
        'bcc_triage_spec': bcc_res['specificity'],
        'bcc_triage_f1': bcc_res['f1'],
        'bcc_triage_threshold': bcc_res['threshold'],
        'bcc_triage_detected': bcc_res['detected'],
        'bcc_operating_points': bcc_res['operating_points'],

        # Joint Malignancy Screening Metrics (MEL + BCC + AKIEC)
        'malignant_triage_recall': mal_res['sensitivity'],
        'malignant_triage_spec': mal_res['specificity'],
        'malignant_triage_f1': mal_res['f1'],
        'malignant_triage_threshold': mal_res['threshold'],
        'malignant_triage_detected': mal_res['detected'],
        'malignant_operating_points': mal_res['operating_points'],

        'all_preds': all_preds,
        'all_targets': all_targets,
        'all_probs': all_probs,
        'report': report
    }


def train_single_model(
    model_name: str,
    args,
    output_dir: Path,
    hw: dict,
    train_df: pd.DataFrame,
    ham_val_df: pd.DataFrame,
    pad_val_df: pd.DataFrame
) -> dict:
    """Trains a single model architecture through Stage 1 & Stage 2 and performs dual-domain validation."""
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    device = hw['device']
    has_cuda = (device.type == 'cuda')
    precision_dtype = hw['precision_dtype']
    use_scaler = has_cuda and not hw['has_bf16']
    scaler = torch.amp.GradScaler('cuda') if use_scaler else None

    # Adaptive Batch Strategy
    micro_batch, grad_accum_steps = compute_adaptive_batch_strategy(
        vram_gb=hw['vram_gb'], model_name=model_name, requested_batch=args.batch_size
    )

    cfg = MODEL_CONFIGS[model_name]
    img_size = args.img_size or cfg['input_size']
    timm_name = cfg['timm_name']

    lr_stage1 = args.lr_stage1 or cfg['default_lr1']
    lr_stage2 = args.lr_stage2 or cfg['default_lr2']
    weight_decay = cfg['weight_decay']

    print(f"\n{'='*80}")
    print(f" [Dual-Domain Multi-Phase Benchmark Pipeline]")
    print(f" Device: {hw['device_name']} ({hw['vram_gb']:.1f} GB VRAM, {hw['device_count']} GPU(s)) | Precision: {hw['precision_name']}")
    print(f" Model: {model_name.upper()} ({timm_name}) | Pretrained: {cfg['pretrained']}")
    print(f" Datasets: Train (HAM10000: {len(train_df)}) | Val In-Domain (HAM: {len(ham_val_df)}) | Val OOD (PAD-UFES: {len(pad_val_df)})")
    print(f"{'='*80}\n")

    model = timm.create_model(timm_name, pretrained=True, num_classes=NUM_CLASSES)
    if hw['device_count'] > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    use_color_constancy = getattr(args, 'color_constancy', False)
    use_tta = getattr(args, 'use_tta', False)

    train_tf_list = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
        transforms.RandomAutocontrast(p=0.3),
        transforms.ToTensor(),
    ]
    val_tf_list = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ]

    if use_color_constancy:
        train_tf_list.append(ShadesOfGray(p=6.0))
        val_tf_list.append(ShadesOfGray(p=6.0))
        print("  🌈 Illumination Constancy: ENABLED (Shades-of-Gray Minkowski p=6.0)")

    train_tf_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    val_tf_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    train_transform = transforms.Compose(train_tf_list)
    val_transform = transforms.Compose(val_tf_list)

    if use_tta:
        print("  🔄 Test-Time Augmentation (TTA): ENABLED (4-view multi-angle inference)")

    balanced_sampling = getattr(args, 'balanced_sampling', False)
    logit_adjust = float(getattr(args, 'logit_adjust', 0.0) or 0.0)
    mixup_minority = float(getattr(args, 'mixup_minority', 0.0) or 0.0)
    train_class_priors = compute_class_priors(train_df['dx'])

    train_ds = SkinDataset(train_df, transform=train_transform)
    ham_val_ds = SkinDataset(ham_val_df, transform=val_transform)
    pad_val_ds = SkinDataset(pad_val_df, transform=val_transform)

    num_workers = min(4, os.cpu_count() or 1)
    use_pin = (has_cuda and model_name != 'v5')

    if balanced_sampling:
        class_counts = train_df['dx'].value_counts().to_dict()
        # Smoothed inverse weighting for balanced representation
        sample_weights = [1.0 / (max(class_counts.get(dx, 1), 1) ** 0.5) for dx in train_df['dx']]
        # Dynamic augmented oversampling: sample enough to expose model to rich variations of all classes
        total_samples = max(class_counts.values()) * NUM_CLASSES
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=total_samples, replacement=True)
        train_loader = DataLoader(train_ds, batch_size=micro_batch, sampler=sampler, num_workers=num_workers, pin_memory=use_pin, drop_last=True)
        print(f"  ⚖️  Dynamic Augmented Oversampling: ENABLED ({total_samples} samples/epoch, drop_last=True)")
    else:
        train_loader = DataLoader(train_ds, batch_size=micro_batch, shuffle=True, num_workers=num_workers, pin_memory=use_pin, drop_last=True)

    if logit_adjust > 0.0:
        print(f"  🎯 Logit Prior Adjustment: ENABLED (tau={logit_adjust:.2f})")
    if mixup_minority > 0.0:
        print(f"  🎨 Minority Class Mixup Augmentation: ENABLED (alpha={mixup_minority:.2f})")

    ham_val_loader = DataLoader(ham_val_ds, batch_size=micro_batch * 2, shuffle=False, num_workers=num_workers, pin_memory=use_pin)
    pad_val_loader = DataLoader(pad_val_ds, batch_size=micro_batch * 2, shuffle=False, num_workers=num_workers, pin_memory=use_pin)

    if balanced_sampling:
        # Batches are already dynamically balanced; use neutral alpha to avoid compounding penalties
        weight_tensor = torch.ones(NUM_CLASSES, dtype=torch.float32).to(device)
    else:
        weights_dict = compute_class_weights(train_df['dx'])
        weight_tensor = torch.tensor([weights_dict[i] for i in range(NUM_CLASSES)], dtype=torch.float32).to(device)
    criterion = PyTorchFocalLoss(alpha=weight_tensor, gamma=2.0)

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 1: Warmup Classifier Head (Backbone Frozen)
    # ──────────────────────────────────────────────────────────────────────────
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    for param in raw_model.parameters():
        param.requires_grad = False
    head = raw_model.get_classifier()
    for param in head.parameters():
        param.requires_grad = True

    stage1_epochs = max(min(args.epochs // 3, 15), 1)
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

            if mixup_minority > 0.0 and model.training:
                inputs_m, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=mixup_minority)
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
            logit_adjust=logit_adjust, class_priors=train_class_priors
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

    # Intermediate Stage 1 benchmark
    s1_ham_eval = evaluate_dataset(model, ham_val_loader, device, precision_dtype, has_cuda, logit_adjust=logit_adjust, class_priors=train_class_priors)
    s1_pad_eval = evaluate_dataset(model, pad_val_loader, device, precision_dtype, has_cuda, logit_adjust=logit_adjust, class_priors=train_class_priors)
    print(f"\n[Stage 1 Complete] In-Domain (HAM) Acc: {s1_ham_eval['accuracy']:.2%} | OOD (PAD-UFES) Acc: {s1_pad_eval['accuracy']:.2%}\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 2: End-to-End Fine-Tuning
    # ──────────────────────────────────────────────────────────────────────────
    if model_name == 'v5':
        # Unfreeze deep semantic stages and adapters for 300M Gemma3n vision encoder
        for name, p in raw_model.named_parameters():
            if any(k in name for k in ['blocks.2', 'blocks.3', 'msfa', 'head', 'classifier']):
                p.requires_grad = True
            else:
                p.requires_grad = False
    else:
        for param in raw_model.parameters():
            param.requires_grad = True

    optimizer2 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_stage2, weight_decay=weight_decay)
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='min', factor=0.3, patience=3, min_lr=1e-8)

    print(f"--- Stage 2: Backbone Fine-Tuning (lr={lr_stage2}, epochs={args.epochs}, patience={args.patience}) ---")
    h2 = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': [], 'time_per_epoch': []}
    best_val_loss = float('inf')
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

            if mixup_minority > 0.0 and model.training:
                inputs_m, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=mixup_minority)
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
            if (step + 1) % 50 == 0 and has_cuda:
                torch.cuda.empty_cache()

        epoch_time = perf_counter() - t0
        train_acc = correct / total
        train_loss = total_loss / total

        val_eval = evaluate_dataset(
            model, ham_val_loader, device, precision_dtype, has_cuda, criterion=criterion,
            logit_adjust=logit_adjust, class_priors=train_class_priors
        )
        val_acc, val_loss = val_eval['accuracy'], val_eval['loss']
        scheduler2.step(val_loss)

        h2['accuracy'].append(train_acc)
        h2['val_accuracy'].append(val_acc)
        h2['loss'].append(train_loss)
        h2['val_loss'].append(val_loss)
        h2['time_per_epoch'].append(epoch_time)

        print(f"  Stage 2 Epoch {epoch+1:02d}/{args.epochs:02d} [{epoch_time:.1f}s] "
              f"— Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
              f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
              f"| lr: {optimizer2.param_groups[0]['lr']:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(raw_model.state_dict(), checkpoint_path)
            print(f"  ⭐ New best checkpoint saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[EarlyStopping] Validation loss did not improve for {args.patience} epochs. Stopping.")
                break

    if checkpoint_path.exists():
        raw_model.load_state_dict(torch.load(checkpoint_path))

    print(f"\n{'='*80}")
    print(f" 🏆 Running Final Dual-Domain Benchmark Evaluation on {model_name.upper()}")
    print(f"{'='*80}\n")

    mel_th = getattr(args, 'mel_threshold', None)
    bcc_th = getattr(args, 'bcc_threshold', 'youden')
    mal_th = getattr(args, 'malignant_threshold', None)

    # 1. In-Domain Evaluation (HAM10000)
    ham_results = evaluate_dataset(
        model, ham_val_loader, device, precision_dtype, has_cuda,
        mel_threshold=mel_th, bcc_threshold=bcc_th, malignant_threshold=mal_th,
        logit_adjust=logit_adjust, class_priors=train_class_priors,
        use_tta=use_tta
    )
    ham_dir = model_dir / 'ham10000'
    ham_dir.mkdir(parents=True, exist_ok=True)
    with open(ham_dir / 'classification_report.json', 'w') as f:
        json.dump(ham_results['report'], f, indent=2)
    plot_confusion_matrices(ham_results['all_targets'], ham_results['all_preds'], CLASS_NAMES, ham_dir / 'confusion_matrix.png', model_name=f"{model_name} (HAM10000)")
    plot_per_class_metrics(ham_results['report'], CLASS_NAMES, ham_dir / 'per_class_metrics.png', model_name=f"{model_name} (HAM10000)")
    plot_roc_curves(ham_results['all_targets'], ham_results['all_probs'], CLASS_NAMES, ham_dir / 'roc_curves.png', model_name=f"{model_name} (HAM10000)")
    generate_gradcam_gallery(model=model, val_df=ham_val_df, class_names=CLASS_NAMES, img_size=img_size, output_path=ham_dir / 'gradcam_heatmaps.png', model_name=f"{model_name}_HAM", device=device)

    # 2. Out-of-Domain Evaluation (PAD-UFES-20)
    calibrated_th = ham_results['mel_triage_threshold']
    calibrated_bcc_th = ham_results['bcc_triage_threshold']
    calibrated_mal_th = ham_results['malignant_triage_threshold']
    pad_results = evaluate_dataset(
        model, pad_val_loader, device, precision_dtype, has_cuda,
        mel_threshold=calibrated_th, bcc_threshold=calibrated_bcc_th, malignant_threshold=calibrated_mal_th,
        logit_adjust=logit_adjust, class_priors=train_class_priors,
        use_tta=use_tta
    )
    pad_dir = model_dir / 'pad_ufes_20'
    pad_dir.mkdir(parents=True, exist_ok=True)
    with open(pad_dir / 'classification_report.json', 'w') as f:
        json.dump(pad_results['report'], f, indent=2)
    plot_confusion_matrices(pad_results['all_targets'], pad_results['all_preds'], CLASS_NAMES, pad_dir / 'confusion_matrix.png', model_name=f"{model_name} (PAD-UFES-20)")
    plot_per_class_metrics(pad_results['report'], CLASS_NAMES, pad_dir / 'per_class_metrics.png', model_name=f"{model_name} (PAD-UFES-20)")
    plot_roc_curves(pad_results['all_targets'], pad_results['all_probs'], CLASS_NAMES, pad_dir / 'roc_curves.png', model_name=f"{model_name} (PAD-UFES-20)")
    generate_gradcam_gallery(model=model, val_df=pad_val_df, class_names=CLASS_NAMES, img_size=img_size, output_path=pad_dir / 'gradcam_heatmaps.png', model_name=f"{model_name}_PAD", device=device)

    # 3. Overall Curves & Domain Comparison Chart
    plot_training_curves([h1, h2], ['Stage 1 (Head)', 'Stage 2 (Fine-Tune)'], model_dir / 'training_curves.png', model_name=model_name)
    plot_domain_comparison(ham_results, pad_results, model_name=model_name, output_path=model_dir / 'domain_comparison.png')
    plot_dual_roc_comparison(ham_results, pad_results, CLASS_NAMES, model_dir / 'roc_curves_dual_domain.png', model_name=model_name)

    # Also save top-level confusion matrix & report (for PAD-UFES-20) for backward compatibility
    plot_confusion_matrices(pad_results['all_targets'], pad_results['all_preds'], CLASS_NAMES, model_dir / 'confusion_matrix.png', model_name=model_name)
    plot_per_class_metrics(pad_results['report'], CLASS_NAMES, model_dir / 'per_class_metrics.png', model_name=model_name)
    plot_roc_curves(pad_results['all_targets'], pad_results['all_probs'], CLASS_NAMES, model_dir / 'roc_curves.png', model_name=model_name)
    generate_gradcam_gallery(model=model, val_df=pad_val_df, class_names=CLASS_NAMES, img_size=img_size, output_path=model_dir / 'gradcam_heatmaps.png', model_name=model_name, device=device)

    domain_gap = ham_results['accuracy'] - pad_results['accuracy']

    full_results = {
        'model': model_name,
        'pretrained': cfg['pretrained'],
        'img_size': img_size,
        'batch_size': args.batch_size,
        'params': sum(p.numel() for p in model.parameters()),
        'mel_triage_threshold': calibrated_th,
        'bcc_triage_threshold': calibrated_bcc_th,
        'malignant_triage_threshold': calibrated_mal_th,
        'balanced_sampling': balanced_sampling,
        'logit_adjust': logit_adjust,
        'mixup_minority': mixup_minority,
        'color_constancy': use_color_constancy,
        'use_tta': use_tta,

        # Stage 1 Benchmarks
        'stage1_ham_accuracy': s1_ham_eval['accuracy'],
        'stage1_pad_accuracy': s1_pad_eval['accuracy'],

        # Stage 2 In-Domain HAM10000 Metrics
        'ham_accuracy': ham_results['accuracy'],
        'ham_weighted_f1': ham_results['weighted_avg_f1'],
        'ham_macro_f1': ham_results['macro_avg_f1'],
        'ham_mel_recall': ham_results['mel_recall'],
        'ham_bcc_recall': ham_results['bcc_recall'],
        'ham_akiec_recall': ham_results['akiec_recall'],
        'ham_mel_auc_roc': ham_results['mel_auc_roc'],
        'ham_macro_auc_roc': ham_results['macro_auc_roc'],
        'ham_harmonized_5class_acc': ham_results['harmonized_5class_acc'],
        'ham_mel_triage_recall': ham_results['mel_triage_recall'],
        'ham_mel_triage_spec': ham_results['mel_triage_spec'],
        'ham_mel_operating_points': ham_results['mel_operating_points'],
        'ham_bcc_triage_recall': ham_results['bcc_triage_recall'],
        'ham_bcc_triage_spec': ham_results['bcc_triage_spec'],
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
        'pad_macro_auc_roc': pad_results['macro_auc_roc'],
        'pad_harmonized_5class_acc': pad_results['harmonized_5class_acc'],
        'pad_mel_triage_recall': pad_results['mel_triage_recall'],
        'pad_mel_triage_spec': pad_results['mel_triage_spec'],
        'pad_mel_triage_detected': pad_results['mel_triage_detected'],
        'pad_mel_operating_points': pad_results['mel_operating_points'],
        'pad_bcc_triage_recall': pad_results['bcc_triage_recall'],
        'pad_bcc_triage_spec': pad_results['bcc_triage_spec'],
        'pad_bcc_triage_detected': pad_results['bcc_triage_detected'],
        'pad_bcc_operating_points': pad_results['bcc_operating_points'],
        'pad_malignant_triage_recall': pad_results['malignant_triage_recall'],
        'pad_malignant_triage_spec': pad_results['malignant_triage_spec'],
        'pad_malignant_triage_detected': pad_results['malignant_triage_detected'],
        'pad_malignant_operating_points': pad_results['malignant_operating_points'],

        # Clinical Domain Shift Drop (In-Domain - Out-of-Domain)
        'domain_gap': float(domain_gap),
        'domain_gap_drop': float(domain_gap)
    }

    with open(model_dir / 'results.json', 'w') as f:
        json.dump(full_results, f, indent=2)

    print(f"\n===========================================================================")
    print(f" 📊 Final Results Summary for {model_name.upper()}:")
    print(f"   In-Domain (HAM10000):     Acc={ham_results['accuracy']:.2%} | Macro AUC={ham_results['macro_auc_roc']:.4f} | Mel AUC={ham_results['mel_auc_roc']:.4f}")
    print(f"     ↳ Argmax:               Mel Recall={ham_results['mel_recall']:.2%} | BCC Recall={ham_results['bcc_recall']:.2%}")
    print(f"     ↳ MEL Triage (tau={calibrated_th:.2f}): Mel Recall={ham_results['mel_triage_recall']:.2%} | Spec={ham_results['mel_triage_spec']:.2%}")
    print(f"     ↳ BCC Triage (tau={calibrated_bcc_th:.2f}): BCC Recall={ham_results['bcc_triage_recall']:.2%} | Spec={ham_results['bcc_triage_spec']:.2%}")
    print(f"     ↳ Malignancy Screen:    Recall={ham_results['malignant_triage_recall']:.2%} | Spec={ham_results['malignant_triage_spec']:.2%}")
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
