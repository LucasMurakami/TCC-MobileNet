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

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

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


CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
NUM_CLASSES = len(CLASS_NAMES)

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
        img = Image.open(self.paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


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


def evaluate_dataset(model, loader, device, precision_dtype, has_cuda):
    """Runs a complete evaluation pass on a dataset loader."""
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    v_loss, v_corr, v_tot = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=precision_dtype) if has_cuda else torch.nullcontext():
                out = model(x)
                loss = criterion(out, y)
            probs = torch.softmax(out.float(), dim=1)
            preds = out.argmax(dim=1)
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

    # ── AUC-ROC Calculations ──
    # 1. Binary Melanoma Triage (Melanoma vs Non-Melanoma)
    mel_idx = CLASS_NAMES.index('mel')
    binary_mel_targets = (all_targets == mel_idx).astype(int)
    mel_probs = all_probs[:, mel_idx]
    
    try:
        if len(np.unique(binary_mel_targets)) > 1:
            mel_auc_roc = float(roc_auc_score(binary_mel_targets, mel_probs))
        else:
            mel_auc_roc = 0.0
    except Exception:
        mel_auc_roc = 0.0

    # 2. Multi-Class One-vs-Rest AUC-ROC
    per_class_auc = {}
    valid_aucs = []
    present_classes = np.unique(all_targets)
    for idx, name in enumerate(CLASS_NAMES):
        if idx in present_classes:
            bin_y = (all_targets == idx).astype(int)
            if len(np.unique(bin_y)) > 1:
                try:
                    cls_auc = float(roc_auc_score(bin_y, all_probs[:, idx]))
                    per_class_auc[name] = cls_auc
                    valid_aucs.append(cls_auc)
                except Exception:
                    per_class_auc[name] = 0.0
            else:
                per_class_auc[name] = 0.0
        else:
            per_class_auc[name] = None
            
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
        'all_preds': all_preds,
        'all_targets': all_targets,
        'all_probs': all_probs,
        'report': report
    }


def main():
    parser = argparse.ArgumentParser(description='Dual-Domain PyTorch Pretrained Models Trainer')
    parser.add_argument('--model', type=str, required=True, choices=['v1', 'v2', 'v3', 'v3small', 'v3large', 'v4', 'v4conv', 'v4convl', 'v5'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr-stage1', type=float, default=None)
    parser.add_argument('--lr-stage2', type=float, default=None)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--img-size', type=int, default=None)
    parser.add_argument('--train-csv', type=str, default=None, help='Training set CSV (HAM10000 80%)')
    parser.add_argument('--val-csv', type=str, default=None, help='In-domain validation set CSV (HAM10000 20%)')
    parser.add_argument('--external-val-csv', type=str, default=None, help='Out-of-domain validation set CSV (PAD-UFES-20)')
    parser.add_argument('--val-dataset', type=str, default='both', choices=['ham10000', 'pad-ufes-20', 'both'])
    parser.add_argument('--output-dir', type=str, default='./mobilenet_outputs')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Canonicalize model aliases to standard 1-per-generation IDs (v1, v2, v3, v4, v5)
    if args.model in ('v4conv', 'v4convl'):
        args.model = 'v4'
    elif args.model in ('v3large', 'v3small'):
        args.model = 'v3'

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    model_dir = output_dir / args.model
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Hardware Detection
    hw = configure_hardware_environment()
    device = hw['device']
    has_cuda = (device.type == 'cuda')
    precision_dtype = hw['precision_dtype']
    use_scaler = has_cuda and not hw['has_bf16']

    # 2. Adaptive Batch Strategy
    micro_batch, grad_accum_steps = compute_adaptive_batch_strategy(
        vram_gb=hw['vram_gb'], model_name=args.model, requested_batch=args.batch_size
    )

    # 3. Load Dataset DataFrames
    data_cache = Path('./data_cache')
    prepared_dir = Path('./dataset_treino')

    if args.train_csv and Path(args.train_csv).exists():
        train_df = pd.read_csv(args.train_csv)
    else:
        train_df = pd.read_csv('./mobilenet_outputs/train_df.csv')

    # In-domain HAM10000 validation
    if args.val_csv and Path(args.val_csv).exists():
        ham_val_df = pd.read_csv(args.val_csv)
    elif (output_dir / 'ham_val_df.csv').exists():
        ham_val_df = pd.read_csv(output_dir / 'ham_val_df.csv')
    else:
        from dataset import prepare_dataset
        _, ham_val_df = prepare_dataset(data_cache, prepared_dir, random_state=args.seed)

    # Out-of-domain PAD-UFES-20 validation
    if args.external_val_csv and Path(args.external_val_csv).exists():
        pad_val_df = pd.read_csv(args.external_val_csv)
    elif (output_dir / 'val_df.csv').exists() and len(pd.read_csv(output_dir / 'val_df.csv')) > 2200:
        pad_val_df = pd.read_csv(output_dir / 'val_df.csv')
    else:
        from dataset import ensure_pad_ufes20_download, load_pad_ufes20_validation
        pad_dir = ensure_pad_ufes20_download(data_cache)
        pad_val_df = load_pad_ufes20_validation(pad_dir)

    cfg = MODEL_CONFIGS[args.model]
    img_size = args.img_size or cfg['input_size']
    timm_name = cfg['timm_name']

    lr_stage1 = args.lr_stage1 or cfg['default_lr1']
    lr_stage2 = args.lr_stage2 or cfg['default_lr2']
    weight_decay = cfg['weight_decay']

    print(f"\n{'='*80}")
    print(f" [Dual-Domain Multi-Phase Benchmark Pipeline]")
    print(f" Device: {hw['device_name']} ({hw['vram_gb']:.1f} GB VRAM, {hw['device_count']} GPU(s)) | Precision: {hw['precision_name']}")
    print(f" Model: {args.model.upper()} ({timm_name}) | Pretrained: {cfg['pretrained']}")
    print(f" Datasets: Train (HAM10000: {len(train_df)}) | Val In-Domain (HAM: {len(ham_val_df)}) | Val OOD (PAD-UFES: {len(pad_val_df)})")
    print(f"{'='*80}\n")

    model = timm.create_model(timm_name, pretrained=True, num_classes=NUM_CLASSES)
    if hw['device_count'] > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    scaler = torch.amp.GradScaler('cuda') if use_scaler else None

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(SkinDataset(train_df, train_transform), batch_size=micro_batch, shuffle=True, num_workers=4, pin_memory=has_cuda)
    ham_val_loader = DataLoader(SkinDataset(ham_val_df, val_transform), batch_size=micro_batch, shuffle=False, num_workers=4, pin_memory=has_cuda)
    pad_val_loader = DataLoader(SkinDataset(pad_val_df, val_transform), batch_size=micro_batch, shuffle=False, num_workers=4, pin_memory=has_cuda)

    weights_dict = compute_class_weights(train_df['dx'])
    weight_tensor = torch.tensor([weights_dict[i] for i in range(NUM_CLASSES)], dtype=torch.float, device=device)
    criterion = PyTorchFocalLoss(alpha=weight_tensor, gamma=2.0)

    # ── Stage 1: Freeze Backbone & Warmup Head ──
    for param in model.parameters():
        param.requires_grad = False
    base_model = model.module if hasattr(model, 'module') else model
    head = base_model.get_classifier() if hasattr(base_model, 'get_classifier') else getattr(base_model, 'head', None)
    if isinstance(head, nn.Module):
        for param in head.parameters():
            param.requires_grad = True

    stage1_epochs = min(5, max(1, args.epochs // 5))
    optimizer1 = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr_stage1, weight_decay=weight_decay)
    print(f"--- Stage 1: Freeze backbone & Warmup head (lr={lr_stage1}, {stage1_epochs} epochs) ---")

    h1 = {'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': []}
    log_interval_s1 = max(1, len(train_loader) // 100)
    for epoch in range(1, stage1_epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        t0 = perf_counter()
        pbar = tqdm(
            train_loader,
            desc=f"  [stage1] Epoch {epoch}/{stage1_epochs}",
            unit="batch",
            file=sys.stdout,
            dynamic_ncols=True,
            miniters=log_interval_s1,
            mininterval=1.0
        )
        for i, (x, y) in enumerate(pbar):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if scaler:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    out = model(x)
                    loss = criterion(out, y) / grad_accum_steps
                scaler.scale(loss).backward()
                if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                    scaler.unscale_(optimizer1)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer1)
                    scaler.update()
                    optimizer1.zero_grad()
            else:
                with torch.amp.autocast('cuda', dtype=precision_dtype) if has_cuda else torch.nullcontext():
                    out = model(x)
                    loss = criterion(out, y) / grad_accum_steps
                loss.backward()
                if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer1.step()
                    optimizer1.zero_grad()

            total_loss += loss.item() * grad_accum_steps * len(y)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += len(y)
            if (i + 1) % log_interval_s1 == 0 or (i + 1) == len(train_loader):
                pbar.set_postfix({'loss': f"{total_loss / total:.4f}", 'acc': f"{correct / total:.4f}"})

        train_acc = correct / total
        train_loss = total_loss / total

        # Evaluate on in-domain HAM10000
        ham_eval = evaluate_dataset(model, ham_val_loader, device, precision_dtype, has_cuda)
        h1['accuracy'].append(train_acc)
        h1['loss'].append(train_loss)
        h1['val_accuracy'].append(ham_eval['accuracy'])
        h1['val_loss'].append(ham_eval['loss'])

        print(f"  [stage1] Epoch {epoch}/{stage1_epochs} summary ({perf_counter()-t0:.1f}s) | Train Acc: {train_acc:.4f} Loss: {train_loss:.4f} | HAM Val Acc: {ham_eval['accuracy']:.4f}\n")
        sys.stdout.flush()

    # Stage 1 Benchmarks
    print("\n--- Benchmarking Stage 1 (Warmup) State ---")
    s1_ham_eval = evaluate_dataset(model, ham_val_loader, device, precision_dtype, has_cuda)
    s1_pad_eval = evaluate_dataset(model, pad_val_loader, device, precision_dtype, has_cuda)
    print(f"  Stage 1 -> In-Domain HAM10000 Acc: {s1_ham_eval['accuracy']:.2%} | Out-of-Domain PAD-UFES-20 Acc: {s1_pad_eval['accuracy']:.2%}\n")

    # ── Stage 2: Deep Fine-Tuning with Early Stopping ──
    if has_cuda:
        torch.cuda.empty_cache()

    if args.model == 'v5':
        # For V5 (300M foundation backbone), fine-tune top representation stages & MSFA attention (~45M params)
        for name, param in model.named_parameters():
            if any(k in name for k in ('msfa', 'head', 'classifier', 'blocks.4', 'blocks.3', 'blocks.2')):
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer2 = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr_stage2, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='min', factor=0.3, patience=2, min_lr=1e-7)
    best_val_acc = 0.0
    checkpoint_path = model_dir / 'best_model.pth'
    patience_counter = 0
    early_stopping_patience = args.patience

    print(f"--- Stage 2: Deep Fine-Tuning with {hw['precision_name']} (lr={lr_stage2}, {args.epochs} epochs, {trainable_params:,} trainable params) ---")
    h2 = {'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': []}
    log_interval_s2 = max(1, len(train_loader) // 100)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        t0 = perf_counter()
        pbar = tqdm(
            train_loader,
            desc=f"  [stage2] Epoch {epoch}/{args.epochs}",
            unit="batch",
            file=sys.stdout,
            dynamic_ncols=True,
            miniters=log_interval_s2,
            mininterval=1.0
        )
        for i, (x, y) in enumerate(pbar):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if scaler:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    out = model(x)
                    loss = criterion(out, y) / grad_accum_steps
                scaler.scale(loss).backward()
                if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                    scaler.unscale_(optimizer2)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer2)
                    scaler.update()
                    optimizer2.zero_grad()
            else:
                with torch.amp.autocast('cuda', dtype=precision_dtype) if has_cuda else torch.nullcontext():
                    out = model(x)
                    loss = criterion(out, y) / grad_accum_steps
                loss.backward()
                if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer2.step()
                    optimizer2.zero_grad()

            total_loss += loss.item() * grad_accum_steps * len(y)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += len(y)
            if (i + 1) % log_interval_s2 == 0 or (i + 1) == len(train_loader):
                pbar.set_postfix({'loss': f"{total_loss / total:.4f}", 'acc': f"{correct / total:.4f}"})

        train_acc = correct / total
        train_loss = total_loss / total

        # Evaluate on primary validation set (HAM10000 in-domain)
        ham_eval = evaluate_dataset(model, ham_val_loader, device, precision_dtype, has_cuda)
        val_acc = ham_eval['accuracy']
        val_loss = ham_eval['loss']
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_obj = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(save_obj, checkpoint_path)
        else:
            patience_counter += 1

        h2['accuracy'].append(train_acc)
        h2['loss'].append(train_loss)
        h2['val_accuracy'].append(val_acc)
        h2['val_loss'].append(val_loss)
        print(f"  [stage2] Epoch {epoch}/{args.epochs} summary ({perf_counter()-t0:.1f}s) | Train Acc: {train_acc:.4f} Loss: {train_loss:.4f} | HAM Val Acc: {val_acc:.4f} Loss: {val_loss:.4f}\n")
        sys.stdout.flush()

        if patience_counter >= early_stopping_patience:
            print(f"\n[EarlyStopping] In-domain validation accuracy did not improve for {early_stopping_patience} consecutive epochs. Restoring best weights from {checkpoint_path}.")
            break

    # ── Final Dual-Domain Comprehensive Evaluation ──
    if checkpoint_path.exists():
        raw_model = model.module if hasattr(model, 'module') else model
        raw_model.load_state_dict(torch.load(checkpoint_path))

    print(f"\n{'='*80}")
    print(f" 🏆 Running Final Dual-Domain Benchmark Evaluation on {args.model.upper()}")
    print(f"{'='*80}\n")

    # 1. In-Domain Evaluation (HAM10000)
    ham_results = evaluate_dataset(model, ham_val_loader, device, precision_dtype, has_cuda)
    ham_dir = model_dir / 'ham10000'
    ham_dir.mkdir(parents=True, exist_ok=True)
    with open(ham_dir / 'classification_report.json', 'w') as f:
        json.dump(ham_results['report'], f, indent=2)
    plot_confusion_matrices(ham_results['all_targets'], ham_results['all_preds'], CLASS_NAMES, ham_dir / 'confusion_matrix.png', model_name=f"{args.model} (HAM10000)")
    plot_per_class_metrics(ham_results['report'], CLASS_NAMES, ham_dir / 'per_class_metrics.png', model_name=f"{args.model} (HAM10000)")
    plot_roc_curves(ham_results['all_targets'], ham_results['all_probs'], CLASS_NAMES, ham_dir / 'roc_curves.png', model_name=f"{args.model} (HAM10000)")
    generate_gradcam_gallery(model=model, val_df=ham_val_df, class_names=CLASS_NAMES, img_size=img_size, output_path=ham_dir / 'gradcam_heatmaps.png', model_name=f"{args.model}_HAM", device=device)

    # 2. Out-of-Domain Evaluation (PAD-UFES-20)
    pad_results = evaluate_dataset(model, pad_val_loader, device, precision_dtype, has_cuda)
    pad_dir = model_dir / 'pad_ufes_20'
    pad_dir.mkdir(parents=True, exist_ok=True)
    with open(pad_dir / 'classification_report.json', 'w') as f:
        json.dump(pad_results['report'], f, indent=2)
    plot_confusion_matrices(pad_results['all_targets'], pad_results['all_preds'], CLASS_NAMES, pad_dir / 'confusion_matrix.png', model_name=f"{args.model} (PAD-UFES-20)")
    plot_per_class_metrics(pad_results['report'], CLASS_NAMES, pad_dir / 'per_class_metrics.png', model_name=f"{args.model} (PAD-UFES-20)")
    plot_roc_curves(pad_results['all_targets'], pad_results['all_probs'], CLASS_NAMES, pad_dir / 'roc_curves.png', model_name=f"{args.model} (PAD-UFES-20)")
    generate_gradcam_gallery(model=model, val_df=pad_val_df, class_names=CLASS_NAMES, img_size=img_size, output_path=pad_dir / 'gradcam_heatmaps.png', model_name=f"{args.model}_PAD", device=device)

    # 3. Overall Curves & Domain Comparison Chart
    plot_training_curves([h1, h2], ['Stage 1 (Head)', 'Stage 2 (Fine-Tune)'], model_dir / 'training_curves.png', model_name=args.model)
    plot_domain_comparison(ham_results, pad_results, model_name=args.model, output_path=model_dir / 'domain_comparison.png')
    plot_dual_roc_comparison(ham_results, pad_results, CLASS_NAMES, model_dir / 'roc_curves_dual_domain.png', model_name=args.model)

    # Also save top-level confusion matrix & report (for PAD-UFES-20) for backward compatibility
    plot_confusion_matrices(pad_results['all_targets'], pad_results['all_preds'], CLASS_NAMES, model_dir / 'confusion_matrix.png', model_name=args.model)
    plot_per_class_metrics(pad_results['report'], CLASS_NAMES, model_dir / 'per_class_metrics.png', model_name=args.model)
    plot_roc_curves(pad_results['all_targets'], pad_results['all_probs'], CLASS_NAMES, model_dir / 'roc_curves.png', model_name=args.model)
    generate_gradcam_gallery(model=model, val_df=pad_val_df, class_names=CLASS_NAMES, img_size=img_size, output_path=model_dir / 'gradcam_heatmaps.png', model_name=args.model, device=device)

    domain_gap = ham_results['accuracy'] - pad_results['accuracy']

    full_results = {
        'model': args.model,
        'pretrained': cfg['pretrained'],
        'img_size': img_size,
        'batch_size': args.batch_size,
        'params': sum(p.numel() for p in model.parameters()),

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

        # Stage 2 Out-of-Domain PAD-UFES-20 Metrics (Explicit & Standardized)
        'pad_accuracy': pad_results['accuracy'],
        'pad_weighted_f1': pad_results['weighted_avg_f1'],
        'pad_macro_f1': pad_results['macro_avg_f1'],
        'pad_mel_recall': pad_results['mel_recall'],
        'pad_bcc_recall': pad_results['bcc_recall'],
        'pad_akiec_recall': pad_results['akiec_recall'],
        'pad_mel_auc_roc': pad_results['mel_auc_roc'],
        'pad_macro_auc_roc': pad_results['macro_auc_roc'],
        'pad_harmonized_5class_acc': pad_results['harmonized_5class_acc'],

        # Out-of-Domain Legacy Aliases (for backward compatibility)
        'accuracy': pad_results['accuracy'],
        'weighted_avg_f1': pad_results['weighted_avg_f1'],
        'macro_avg_f1': pad_results['macro_avg_f1'],
        'mel_recall': pad_results['mel_recall'],
        'bcc_recall': pad_results['bcc_recall'],
        'akiec_recall': pad_results['akiec_recall'],
        'mel_auc_roc': pad_results['mel_auc_roc'],
        'macro_auc_roc': pad_results['macro_auc_roc'],
        'harmonized_5class_acc': pad_results['harmonized_5class_acc'],

        # Clinical Domain Shift Drop (In-Domain - Out-of-Domain)
        'domain_gap': float(domain_gap),
        'domain_gap_drop': float(domain_gap)
    }

    with open(model_dir / 'results.json', 'w') as f:
        json.dump(full_results, f, indent=2)

    print(f"\n===========================================================================")
    print(f" 📊 Final Results Summary for {args.model.upper()}:")
    print(f"   In-Domain (HAM10000):   Accuracy={ham_results['accuracy']:.2%} | Weighted F1={ham_results['weighted_avg_f1']:.4f} | Mel Recall={ham_results['mel_recall']:.2%} | Mel AUC-ROC={ham_results['mel_auc_roc']:.4f}")
    print(f"   Out-of-Domain (PAD-UFES): Accuracy={pad_results['accuracy']:.2%} | Weighted F1={pad_results['weighted_avg_f1']:.4f} | Mel Recall={pad_results['mel_recall']:.2%} | Mel AUC-ROC={pad_results['mel_auc_roc']:.4f}")
    print(f"   Clinical Domain Gap (Δ Acc Drop): -{domain_gap*100:.2f}%")
    print(f"   Results saved to: {model_dir / 'results.json'}")
    print(f"   Domain Comparison Chart: {model_dir / 'domain_comparison.png'}")
    print(f"   Dual-Domain ROC Analysis: {model_dir / 'roc_curves_dual_domain.png'}")
    print(f"===========================================================================\n")


if __name__ == '__main__':
    main()
