"""
PyTorch & timm Pretrained Models Trainer (MobileNet V1 - V5)
Modular Hardware Accelerator Engine supporting all GPU architectures:
- NVIDIA Blackwell (RTX 50xx), Ada Lovelace (RTX 40xx), Ampere (RTX 30xx, A100), Hopper (H100)
- NVIDIA Turing (RTX 20xx, T4), Volta (V100), Pascal (GTX 10xx)
- Multi-GPU DataParallel scaling
- Apple Silicon (MPS) and CPU fallback
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
from sklearn.metrics import classification_report, confusion_matrix

torch.backends.cudnn.enabled = False

# ─── Modular Hardware Engine ────────────────────────────────────────────────

def configure_hardware_environment() -> dict:
    """
    Detects GPU architecture, compute capability, VRAM size, native BF16/FP16 support,
    and safely configures CuDNN and memory allocation.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        device_count = torch.cuda.device_count()

        # BF16 support: Ampere (sm_80+), Ada (sm_89), Hopper (sm_90), Blackwell (sm_120)
        has_bf16 = torch.cuda.is_bf16_supported()
        precision_dtype = torch.bfloat16 if has_bf16 else torch.float16
        precision_name = 'BFloat16 (BF16)' if has_bf16 else 'Float16 (FP16)'

        # Test CuDNN compatibility safely
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
    """
    Computes optimal physical micro-batch size and gradient accumulation steps
    adapted to the available VRAM and architecture parameter count.
    """
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


# Canonical classes
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
NUM_CLASSES = len(CLASS_NAMES)
HAM10000_CLASSES = {
    'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6
}
PAD_UFES20_LABEL_MAP = {
    'ACK': 'akiec', 'BCC': 'bcc', 'SEK': 'bkl', 'MEL': 'mel', 'NEV': 'nv', 'SCC': 'akiec'
}

MODEL_CONFIGS = {
    'v1':      {'timm_name': 'mobilenetv1_100',                        'input_size': 224, 'default_lr1': 1e-3, 'default_lr2': 1e-4, 'weight_decay': 1e-4, 'pretrained': 'ImageNet-1k (timm)'},
    'v2':      {'timm_name': 'mobilenetv2_100',                        'input_size': 224, 'default_lr1': 1e-3, 'default_lr2': 1e-4, 'weight_decay': 1e-4, 'pretrained': 'ImageNet-1k (timm)'},
    'v3small': {'timm_name': 'mobilenetv3_small_100',                  'input_size': 224, 'default_lr1': 1e-3, 'default_lr2': 1e-4, 'weight_decay': 1e-4, 'pretrained': 'ImageNet-1k (timm)'},
    'v3large': {'timm_name': 'mobilenetv3_large_100',                  'input_size': 224, 'default_lr1': 1e-3, 'default_lr2': 1e-4, 'weight_decay': 1e-4, 'pretrained': 'ImageNet-1k (timm)'},
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


def main():
    parser = argparse.ArgumentParser(description='Modular PyTorch/timm Pretrained Models Trainer')
    parser.add_argument('--model', type=str, required=True, choices=['v1', 'v2', 'v3', 'v3small', 'v3large', 'v4', 'v4conv', 'v4convl', 'v5'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr-stage1', type=float, default=None)
    parser.add_argument('--lr-stage2', type=float, default=None)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--img-size', type=int, default=None)
    parser.add_argument('--train-csv', type=str, default=None, help='Path to train dataset CSV')
    parser.add_argument('--val-csv', type=str, default=None, help='Path to val dataset CSV')
    parser.add_argument('--val-dataset', type=str, default='ham10000', choices=['ham10000', 'pad-ufes-20'])
    parser.add_argument('--output-dir', type=str, default='./mobilenet_outputs')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.model == 'v4':
        args.model = 'v4conv'
    elif args.model == 'v3':
        args.model = 'v3large'

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    model_dir = output_dir / args.model
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Modular Hardware Detection
    hw = configure_hardware_environment()
    device = hw['device']
    has_cuda = (device.type == 'cuda')
    precision_dtype = hw['precision_dtype']
    use_scaler = has_cuda and not hw['has_bf16']

    # 2. Adaptive Batch Sizing
    micro_batch, grad_accum_steps = compute_adaptive_batch_strategy(
        vram_gb=hw['vram_gb'], model_name=args.model, requested_batch=args.batch_size
    )

    if args.train_csv and Path(args.train_csv).exists():
        train_df = pd.read_csv(args.train_csv)
    else:
        train_df = pd.read_csv(output_dir / 'train_df.csv')

    if args.val_csv and Path(args.val_csv).exists():
        val_df = pd.read_csv(args.val_csv)
    else:
        val_df = pd.read_csv(output_dir / 'val_df.csv')

    cfg = MODEL_CONFIGS[args.model]
    img_size = args.img_size or cfg['input_size']
    timm_name = cfg['timm_name']

    lr_stage1 = args.lr_stage1 or cfg['default_lr1']
    lr_stage2 = args.lr_stage2 or cfg['default_lr2']
    weight_decay = cfg['weight_decay']

    print(f"\n{'='*75}")
    print(f" [Modular Accelerator Engine]")
    print(f" Device: {hw['device_name']} ({hw['vram_gb']:.1f} GB VRAM, {hw['device_count']} GPU(s))")
    print(f" Precision: {hw['precision_name']} | Target Batch: {args.batch_size} (Micro-batch: {micro_batch}, Accum: {grad_accum_steps})")
    print(f" Model: {args.model.upper()} ({timm_name}) | Pretrained: {cfg['pretrained']}")
    print(f" Hyperparameters: LR Stage1={lr_stage1}, LR Stage2={lr_stage2}, WeightDecay={weight_decay}")
    print(f"{'='*75}")

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
    val_loader = DataLoader(SkinDataset(val_df, val_transform), batch_size=micro_batch, shuffle=False, num_workers=4, pin_memory=has_cuda)

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
    print(f"\n--- Stage 1: Freeze backbone & Warmup head (lr={lr_stage1}, {stage1_epochs} epochs) ---")

    h1 = {'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': []}
    for epoch in range(1, stage1_epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        t0 = perf_counter()
        pbar = tqdm(train_loader, desc=f"  [stage1] Epoch {epoch}/{stage1_epochs}", unit="batch", file=sys.stdout, dynamic_ncols=True)
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
            pbar.set_postfix({'loss': f"{total_loss / total:.4f}", 'acc': f"{correct / total:.4f}"})

        train_acc = correct / total
        train_loss = total_loss / total

        model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=precision_dtype) if has_cuda else torch.nullcontext():
                    out = model(x)
                    loss = criterion(out, y)
                v_loss += loss.item() * len(y)
                v_corr += (out.argmax(dim=1) == y).sum().item()
                v_tot += len(y)
        val_acc = v_corr / v_tot
        val_loss = v_loss / v_tot

        h1['accuracy'].append(train_acc)
        h1['loss'].append(train_loss)
        h1['val_accuracy'].append(val_acc)
        h1['val_loss'].append(val_loss)
        print(f"  [stage1] Epoch {epoch}/{stage1_epochs} summary ({perf_counter()-t0:.1f}s) | Train Acc: {train_acc:.4f} Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} Loss: {val_loss:.4f}\n")
        sys.stdout.flush()

    # ── Stage 2: Deep Fine-Tuning with Early Stopping & AdamW ──
    if has_cuda:
        torch.cuda.empty_cache()

    raw_model = model.module if hasattr(model, 'module') else model
    if hasattr(raw_model, 'set_grad_checkpointing'):
        try:
            raw_model.set_grad_checkpointing(True)
        except Exception:
            pass

    if args.model == 'v5':
        for name, param in raw_model.named_parameters():
            if any(k in name for k in ('msfa', 'head', 'classifier', 'blocks.4', 'blocks.3')):
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

    print(f"\n--- Stage 2: Deep Fine-Tuning with {hw['precision_name']} (lr={lr_stage2}, {args.epochs} epochs, {trainable_params:,} trainable params) ---")
    h2 = {'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': []}
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        t0 = perf_counter()
        pbar = tqdm(train_loader, desc=f"  [stage2] Epoch {epoch}/{args.epochs}", unit="batch", file=sys.stdout, dynamic_ncols=True)
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
            pbar.set_postfix({'loss': f"{total_loss / total:.4f}", 'acc': f"{correct / total:.4f}"})

        train_acc = correct / total
        train_loss = total_loss / total

        model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=precision_dtype) if has_cuda else torch.nullcontext():
                    out = model(x)
                    loss = criterion(out, y)
                v_loss += loss.item() * len(y)
                v_corr += (out.argmax(dim=1) == y).sum().item()
                v_tot += len(y)
        val_acc = v_corr / v_tot
        val_loss = v_loss / v_tot
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
        print(f"  [stage2] Epoch {epoch}/{args.epochs} summary ({perf_counter()-t0:.1f}s) | Train Acc: {train_acc:.4f} Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} Loss: {val_loss:.4f}\n")
        sys.stdout.flush()

        if patience_counter >= early_stopping_patience:
            print(f"\n[EarlyStopping] Validation accuracy did not improve for {early_stopping_patience} consecutive epochs. Restoring best weights from {checkpoint_path}.")
            break

    if checkpoint_path.exists():
        raw_model = model.module if hasattr(model, 'module') else model
        raw_model.load_state_dict(torch.load(checkpoint_path))

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            with torch.amp.autocast('cuda', dtype=precision_dtype) if has_cuda else torch.nullcontext():
                out = model(x)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    report = classification_report(all_targets, all_preds, labels=range(NUM_CLASSES), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    print(f"\nClassification Report for {args.model}:")
    print(classification_report(all_targets, all_preds, labels=range(NUM_CLASSES), target_names=CLASS_NAMES, zero_division=0))

    with open(model_dir / 'classification_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    from visualize import plot_confusion_matrices, plot_per_class_metrics, plot_training_curves, generate_gradcam_gallery

    plot_confusion_matrices(all_targets, all_preds, CLASS_NAMES, model_dir / 'confusion_matrix.png', model_name=args.model)
    plot_per_class_metrics(report, CLASS_NAMES, model_dir / 'per_class_metrics.png', model_name=args.model)
    plot_training_curves([h1, h2], ['Stage 1 (Head)', 'Stage 2 (Fine-Tune)'], model_dir / 'training_curves.png', model_name=args.model)

    print("\nGenerating Grad-CAM CNN attention heatmaps for every diagnostic class...")
    try:
        generate_gradcam_gallery(
            model=model,
            val_df=val_df,
            class_names=CLASS_NAMES,
            img_size=img_size,
            output_path=model_dir / 'gradcam_heatmaps.png',
            model_name=args.model,
            device=device,
        )
        print(f"Grad-CAM gallery saved to: {model_dir / 'gradcam_heatmaps.png'}")
    except Exception as e:
        print(f"Warning: Grad-CAM generation encountered an issue: {e}")

    results = {
        'model': args.model,
        'pretrained': cfg['pretrained'],
        'img_size': img_size,
        'batch_size': args.batch_size,
        'accuracy': float(report['accuracy']),
        'weighted_avg_f1': float(report['weighted avg']['f1-score']),
        'macro_avg_f1': float(report['macro avg']['f1-score']),
        'params': sum(p.numel() for p in model.parameters()),
    }
    with open(model_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {model_dir / 'results.json'}")


if __name__ == '__main__':
    main()
