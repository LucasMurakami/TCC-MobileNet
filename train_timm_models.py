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
from torch.utils.data import Dataset, DataLoader
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


def evaluate_dataset(model, loader, device, precision_dtype, has_cuda, criterion=None):
    """Runs a complete evaluation pass on a dataset loader.
    
    Args:
        criterion: Loss function to use for validation loss computation.
                   If None, defaults to CrossEntropyLoss for standard reporting.
                   Pass the training Focal Loss during training to align
                   validation loss with the training loss landscape.
    """
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    v_loss, v_corr, v_tot = 0.0, 0, 0
    if criterion is None:
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

    train_ds = SkinDataset(train_df, transform=train_transform)
    ham_val_ds = SkinDataset(ham_val_df, transform=val_transform)
    pad_val_ds = SkinDataset(pad_val_df, transform=val_transform)

    num_workers = min(4, os.cpu_count() or 1)
    use_pin = (has_cuda and model_name != 'v5')
    train_loader = DataLoader(train_ds, batch_size=micro_batch, shuffle=True, num_workers=num_workers, pin_memory=use_pin)
    ham_val_loader = DataLoader(ham_val_ds, batch_size=micro_batch * 2, shuffle=False, num_workers=num_workers, pin_memory=use_pin)
    pad_val_loader = DataLoader(pad_val_ds, batch_size=micro_batch * 2, shuffle=False, num_workers=num_workers, pin_memory=use_pin)

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

        val_eval = evaluate_dataset(model, ham_val_loader, device, precision_dtype, has_cuda, criterion=criterion)
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
    s1_ham_eval = evaluate_dataset(model, ham_val_loader, device, precision_dtype, has_cuda)
    s1_pad_eval = evaluate_dataset(model, pad_val_loader, device, precision_dtype, has_cuda)
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

        val_eval = evaluate_dataset(model, ham_val_loader, device, precision_dtype, has_cuda, criterion=criterion)
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

    # 1. In-Domain Evaluation (HAM10000)
    ham_results = evaluate_dataset(model, ham_val_loader, device, precision_dtype, has_cuda)
    ham_dir = model_dir / 'ham10000'
    ham_dir.mkdir(parents=True, exist_ok=True)
    with open(ham_dir / 'classification_report.json', 'w') as f:
        json.dump(ham_results['report'], f, indent=2)
    plot_confusion_matrices(ham_results['all_targets'], ham_results['all_preds'], CLASS_NAMES, ham_dir / 'confusion_matrix.png', model_name=f"{model_name} (HAM10000)")
    plot_per_class_metrics(ham_results['report'], CLASS_NAMES, ham_dir / 'per_class_metrics.png', model_name=f"{model_name} (HAM10000)")
    plot_roc_curves(ham_results['all_targets'], ham_results['all_probs'], CLASS_NAMES, ham_dir / 'roc_curves.png', model_name=f"{model_name} (HAM10000)")
    generate_gradcam_gallery(model=model, val_df=ham_val_df, class_names=CLASS_NAMES, img_size=img_size, output_path=ham_dir / 'gradcam_heatmaps.png', model_name=f"{model_name}_HAM", device=device)

    # 2. Out-of-Domain Evaluation (PAD-UFES-20)
    pad_results = evaluate_dataset(model, pad_val_loader, device, precision_dtype, has_cuda)
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

        # Clinical Domain Shift Drop (In-Domain - Out-of-Domain)
        'domain_gap': float(domain_gap),
        'domain_gap_drop': float(domain_gap)
    }

    with open(model_dir / 'results.json', 'w') as f:
        json.dump(full_results, f, indent=2)

    print(f"\n===========================================================================")
    print(f" 📊 Final Results Summary for {model_name.upper()}:")
    print(f"   In-Domain (HAM10000):   Accuracy={ham_results['accuracy']:.2%} | Weighted F1={ham_results['weighted_avg_f1']:.4f} | Mel Recall={ham_results['mel_recall']:.2%} | Mel AUC-ROC={ham_results['mel_auc_roc']:.4f}")
    print(f"   Out-of-Domain (PAD-UFES): Accuracy={pad_results['accuracy']:.2%} | Weighted F1={pad_results['weighted_avg_f1']:.4f} | Mel Recall={pad_results['mel_recall']:.2%} | Mel AUC-ROC={pad_results['mel_auc_roc']:.4f}")
    print(f"   Clinical Domain Gap (Δ Acc Drop): -{domain_gap*100:.2f}%")
    print(f"   Results saved to: {model_dir / 'results.json'}")
    print(f"   Domain Comparison Chart: {model_dir / 'domain_comparison.png'}")
    print(f"   Dual-Domain ROC Analysis: {model_dir / 'roc_curves_dual_domain.png'}")
    print(f"===========================================================================\n")

    return full_results


if __name__ == '__main__':
    from main import main
    main()
