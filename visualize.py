"""
Advanced Data Visualization & CNN Interpretability Suite for Skin Lesion Classification
Includes:
  - Enhanced Training & Validation Curves (Stage 1 + Stage 2 unified or separate)
  - Dual-View Confusion Matrices (Raw Counts + Normalized Percentages)
  - Per-Class Performance Bar Charts (Precision, Recall, F1)
  - PyTorch Grad-CAM (Class Activation Mapping) CNN Heatmaps per Class
  - Multi-Model Benchmark Comparison Plots
"""

import os
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score

import torch
import torch.nn as nn
from torchvision import transforms

# ─── 1. Training & Validation Curves ──────────────────────────────────────────

def plot_training_curves(
    stages_history: List[Dict[str, List[float]]],
    stage_names: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    model_name: str = "Model"
) -> plt.Figure:
    """Plot publication-quality training & validation curves across multiple stages."""
    if stage_names is None:
        stage_names = [f"Stage {i+1}" for i in range(len(stages_history))]

    all_train_acc, all_val_acc = [], []
    all_train_loss, all_val_loss = [], []

    for h in stages_history:
        all_train_acc.extend(h.get('accuracy', []))
        all_val_acc.extend(h.get('val_accuracy', []))
        all_train_loss.extend(h.get('loss', []))
        all_val_loss.extend(h.get('val_loss', []))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.patch.set_facecolor('#fafafa')
    epochs_range = range(1, len(all_train_acc) + 1)

    # Accuracy Plot
    ax1.set_facecolor('#ffffff')
    ax1.plot(epochs_range, all_train_acc, label='Train Accuracy', color='#1f77b4', lw=2.2, marker='o', markersize=4)
    ax1.plot(epochs_range, all_val_acc, label='Val Accuracy', color='#ff7f0e', lw=2.2, marker='s', markersize=4)
    ax1.set_title(f"{model_name.upper()} — Classification Accuracy", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Accuracy', fontsize=10)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9)

    # Loss Plot
    ax2.set_facecolor('#ffffff')
    ax2.plot(epochs_range, all_train_loss, label='Train Loss', color='#1f77b4', lw=2.2, marker='o', markersize=4)
    ax2.plot(epochs_range, all_val_loss, label='Val Loss', color='#ff7f0e', lw=2.2, marker='s', markersize=4)
    ax2.set_title(f"{model_name.upper()} — Loss (Focal Loss)", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Focal Loss', fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=250, bbox_inches='tight')
        plt.close(fig)
    return fig


# ─── 2. Dual-View Confusion Matrices ──────────────────────────────────────────

def plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    model_name: str = "Model"
) -> plt.Figure:
    """Plot dual confusion matrix: Raw counts & Normalized percentages side-by-side."""
    cm_raw = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm_raw.astype('float') / cm_raw.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#fafafa')

    # Raw counts
    sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar=True, ax=axes[0], annot_kws={"size": 10, "weight": "bold"})
    axes[0].set_title(f"{model_name.upper()} — Sample Counts", fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel('Predicted Diagnostic Class', fontsize=11, labelpad=8)
    axes[0].set_ylabel('True Diagnostic Class', fontsize=11, labelpad=8)

    # Normalized percentages
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='YlGnBu',
                xticklabels=class_names, yticklabels=class_names,
                cbar=True, ax=axes[1], annot_kws={"size": 10, "weight": "bold"})
    axes[1].set_title(f"{model_name.upper()} — Normalized Recall / Sensitivity", fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel('Predicted Diagnostic Class', fontsize=11, labelpad=8)
    axes[1].set_ylabel('True Diagnostic Class', fontsize=11, labelpad=8)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=250, bbox_inches='tight')
        plt.close(fig)
    return fig


# ─── 3. Per-Class Metrics Bar Chart ──────────────────────────────────────────

def plot_per_class_metrics(
    report: Dict[str, Any],
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    model_name: str = "Model"
) -> plt.Figure:
    """Plot per-class Precision, Recall, and F1-Score bar chart with annotations."""
    precisions = [report.get(c, {}).get('precision', 0.0) for c in class_names]
    recalls = [report.get(c, {}).get('recall', 0.0) for c in class_names]
    f1_scores = [report.get(c, {}).get('f1-score', 0.0) for c in class_names]

    x = np.arange(len(class_names))
    width = 0.26

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#ffffff')

    rects1 = ax.bar(x - width, precisions, width, label='Precision', color='#4C72B0', edgecolor='white')
    rects2 = ax.bar(x, recalls, width, label='Recall (Sensitivity)', color='#55A868', edgecolor='white')
    rects3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color='#C44E52', edgecolor='white')

    ax.set_title(f"{model_name.upper()} — Per-Class Diagnostic Performance", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Diagnostic Class', fontsize=11, labelpad=8)
    ax.set_ylabel('Score [0.0 - 1.0]', fontsize=11, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', linestyle=':', alpha=0.7)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

    # Add numeric labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0.01:
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=250, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_reliability_diagram(probs: np.ndarray, targets: np.ndarray, output_path: Union[str, Path], model_name: str = "Model", n_bins: int = 15):
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == targets
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    accuracy = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins)
    for index in range(n_bins):
        mask = (confidence >= edges[index]) & (confidence < edges[index + 1] if index < n_bins - 1 else confidence <= edges[index + 1])
        counts[index] = mask.sum()
        if np.any(mask):
            accuracy[index] = correct[mask].mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray')
    axes[0].plot(centers, accuracy, marker='o', color='#1f77b4')
    axes[0].set(xlabel='Confidence', ylabel='Accuracy', title=f'{model_name} Reliability', xlim=(0, 1), ylim=(0, 1))
    axes[0].grid(alpha=0.25)
    axes[1].bar(centers, counts, width=1 / n_bins * 0.9, color='#1f77b4')
    axes[1].set(xlabel='Confidence', ylabel='Samples', title='Confidence Distribution', xlim=(0, 1))
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# ─── 4. PyTorch Grad-CAM & Grad-CAM++ Engine ─────────────────────────────────

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None


def find_gradcam_target_layer(model: nn.Module) -> Optional[nn.Module]:
    """
    Finds the optimal spatial convolutional layer for Grad-CAM across MobileNet V1, V2, V3, V4, and V5.
    Explicitly targets the deepest spatial convolutional stage (e.g. blocks[-1]) while skipping 1x1 post-pooling
    conv_head layers, Squeeze-and-Excitation (SE) modules, and linear classification heads.
    """
    raw_model = model.module if hasattr(model, 'module') else model

    # Excluded keywords that belong to SE reduction/expansion, linear classifiers, or pooling layers
    excluded_keywords = (
        'classifier', 'fc', 'linear', 'head.fc', 'head.flatten', 'norm_head',
        'se.', '.se.', 'conv_reduce', 'conv_expand', 'se_module', 'squeeze', 'excitation'
    )

    # 0. MobileNetV5 / Gemma3n: target the final pre-normalization spatial convolution
    if hasattr(raw_model, 'msfa'):
        if hasattr(raw_model.msfa, 'ffn') and hasattr(raw_model.msfa.ffn, 'pw_proj'):
            return raw_model.msfa.ffn.pw_proj.conv
        if hasattr(raw_model, 'blocks') and len(raw_model.blocks) >= 4:
            stage = raw_model.blocks[3]
            candidates = [module for module in stage.modules() if isinstance(module, nn.Conv2d)]
            if candidates:
                return candidates[-1]
        return raw_model.msfa

    # 1. Primary priority: The last convolutional layer in blocks[-1] (the final spatial feature extraction stage)
    # This guarantees true 7x7 spatial maps for V3, V4, and modern timm backbones
    if hasattr(raw_model, 'blocks') and len(raw_model.blocks) > 0:
        last_block = raw_model.blocks[-1]
        candidates = []
        for name, mod in last_block.named_modules():
            if isinstance(mod, nn.Conv2d):
                name_lower = name.lower()
                if not any(k in name_lower for k in excluded_keywords):
                    candidates.append(mod)
        if candidates:
            return candidates[-1]
        return last_block

    # 2. V1 / V2 check conv_head if before pooling
    if hasattr(raw_model, 'conv_head') and isinstance(raw_model.conv_head, nn.Conv2d):
        return raw_model.conv_head

    # 3. Check candidate layers across the entire backbone from deepest to shallowest
    candidate_layers = []
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Conv2d):
            name_lower = name.lower()
            if not any(k in name_lower for k in excluded_keywords):
                candidate_layers.append((name, module))

    if candidate_layers:
        return candidate_layers[-1][1]

    # Fallback to any Conv2d
    all_convs = [m for m in raw_model.modules() if isinstance(m, nn.Conv2d)]
    return all_convs[-1] if all_convs else None


class PyTorchGradCAM:
    """
    Robust Grad-CAM / Grad-CAM++ activation mapping engine for PyTorch/timm vision models.
    Uses the final pre-normalization spatial convolution for MobileNetV5 and the deepest convolutional stage for other backbones.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer or find_gradcam_target_layer(model)
        self.activations = None
        self.gradients = None
        self.hook_handle = None
        raw_model = model.module if hasattr(model, 'module') else model

        # Disable in-place activations to guarantee backward gradient flow to hooked outputs
        for module in raw_model.modules():
            if hasattr(module, 'inplace'):
                module.inplace = False

        if self.target_layer is not None:
            self.hook_handle = self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.clone()
        if out.requires_grad:
            out.register_hook(self._tensor_backward_hook)

    def _tensor_backward_hook(self, grad):
        self.gradients = grad.clone()

    def __call__(self, x: torch.Tensor, target_class: Optional[int] = None) -> tuple:
        self.activations = None
        self.gradients = None
        orig_states = [p.requires_grad for p in self.model.parameters()]
        for p in self.model.parameters():
            p.requires_grad = True

        try:
            with torch.enable_grad():
                self.model.eval()
                self.model.zero_grad()
                x_in = x.clone().detach().requires_grad_(True)
                out = self.model(x_in)

                if target_class is None:
                    target_class = int(out.argmax(dim=1).item())

                score = out[0, target_class]
                score.backward(retain_graph=True)

                if self.activations is None or self.gradients is None:
                    # Fallback if layer produces no gradients
                    heatmap_np = np.zeros((x.shape[2], x.shape[3]), dtype=np.float32)
                else:
                    acts = self.activations.float()
                    grads = self.gradients.float()
                    H, W = x.shape[2], x.shape[3]
                    eps = 1e-7

                    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
                    cam = torch.relu(torch.sum(weights * acts, dim=1, keepdim=True))

                    if cam.max() < eps:
                        grads_power_2 = grads.pow(2)
                        grads_power_3 = grads.pow(3)
                        sum_acts = acts.sum(dim=(2, 3), keepdim=True)
                        aij = grads_power_2 / (2.0 * grads_power_2 + sum_acts * grads_power_3 + eps)
                        aij = torch.where(grads != 0, aij, torch.zeros_like(aij))
                        weights_pp = torch.sum(aij * torch.relu(grads), dim=(2, 3), keepdim=True)
                        cam = torch.relu(torch.sum(weights_pp * acts, dim=1, keepdim=True))

                    if cam.max() < eps:
                        cam = torch.mean(acts.abs(), dim=1, keepdim=True)

                    cam = cam - cam.min()
                    max_val = cam.max()
                    if max_val > eps:
                        cam = cam / max_val

                    cam = nn.functional.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
                    heatmap_np = cam[0, 0].detach().cpu().numpy()

                    if gaussian_filter is not None:
                        heatmap_np = gaussian_filter(heatmap_np, sigma=max(H, W) / 200.0)
                        h_max = heatmap_np.max()
                        if h_max > eps:
                            heatmap_np = (heatmap_np - heatmap_np.min()) / (h_max - heatmap_np.min())

                probs_np = out.softmax(dim=1)[0].detach().cpu().numpy()

        finally:
            for p, state in zip(self.model.parameters(), orig_states):
                p.requires_grad = state

        return heatmap_np, probs_np

    def remove_hooks(self):
        if self.hook_handle is not None:
            try:
                self.hook_handle.remove()
            except Exception:
                pass
            self.hook_handle = None


def overlay_gradcam(original_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45, colormap: str = 'jet') -> np.ndarray:
    """Superimpose Grad-CAM heatmap onto RGB image [0, 255]."""
    cmap = matplotlib.colormaps[colormap]
    colored_heatmap = cmap(heatmap)[:, :, :3]
    colored_heatmap = (colored_heatmap * 255).astype(np.uint8)

    if original_img.max() <= 1.0:
        base_img = (original_img * 255).astype(np.uint8)
    else:
        base_img = original_img.astype(np.uint8)

    # Resize base image to match heatmap if needed
    if base_img.shape[:2] != heatmap.shape[:2]:
        base_img = np.array(Image.fromarray(base_img).resize((heatmap.shape[1], heatmap.shape[0])))

    superimposed = (colored_heatmap * alpha + base_img * (1.0 - alpha)).astype(np.uint8)
    return superimposed


def generate_gradcam_gallery(
    model: nn.Module,
    val_df: pd.DataFrame,
    class_names: List[str],
    img_size: int = 224,
    output_path: Optional[Union[str, Path]] = None,
    model_name: str = "Model",
    device: Optional[torch.device] = None,
    target_mode: str = "pred",
    transform=None,
) -> Optional[plt.Figure]:
    """
    Generates a CNN/ViT interpretability Grad-CAM gallery containing at least ONE representative
    sample for EVERY diagnostic class present in the validation dataset.

    Args:
        target_mode: 'pred' (attention on predicted class), 'true' (attention on ground truth class),
                     or 'contrastive' (4 columns: Input | Pred CAM | True Class CAM | Overlay).
    """
    raw_model = model.module if hasattr(model, 'module') else model
    if device is None:
        try:
            device = next(raw_model.parameters()).device
        except (StopIteration, AttributeError):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raw_model = raw_model.to(device)

    cam_gen = PyTorchGradCAM(raw_model)

    # Find at least 1 sample per class
    samples = []
    for cls in class_names:
        matching = val_df[val_df['dx'] == cls]
        if len(matching) > 0:
            samples.append((cls, matching.sample(1, random_state=42).iloc[0]['path']))

    if not samples:
        cam_gen.remove_hooks()
        return None

    num_samples = len(samples)
    is_contrastive = (target_mode == "contrastive")
    num_cols = 4 if is_contrastive else 3
    fig_width = 16 if is_contrastive else 13

    fig, axes = plt.subplots(num_samples, num_cols, figsize=(fig_width, 3.8 * num_samples))
    fig.patch.set_facecolor('#fafafa')

    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    for row_idx, (true_class, img_path) in enumerate(samples):
        try:
            pil_img = Image.open(img_path).convert('RGB')
            img_resized = pil_img.resize((img_size, img_size))
            img_np = np.array(img_resized)
            img_tensor = transform(pil_img).unsqueeze(0).to(device)

            true_label_idx = class_names.index(true_class)

            # Generate CAM based on target_mode
            if target_mode == "true":
                heatmap, probs = cam_gen(img_tensor, target_class=true_label_idx)
            else:
                heatmap, probs = cam_gen(img_tensor, target_class=None)

            pred_label_idx = int(np.argmax(probs))
            pred_class = class_names[pred_label_idx]
            confidence = float(probs[pred_label_idx])
            true_conf = float(probs[true_label_idx])
            superimposed = overlay_gradcam(img_np, heatmap, alpha=0.45, colormap='jet')

            is_correct = (true_label_idx == pred_label_idx)
            status_color = '#2ca02c' if is_correct else '#d62728'

            # 1. Original Image
            axes[row_idx, 0].imshow(img_np)
            axes[row_idx, 0].set_title(f"Input: {Path(img_path).name}\nTrue Class: {true_class.upper()}", fontsize=10, fontweight='bold')
            axes[row_idx, 0].axis('off')

            if is_contrastive:
                # 2. Predicted Class CAM
                axes[row_idx, 1].imshow(heatmap, cmap='jet')
                axes[row_idx, 1].set_title(f"Pred CAM: {pred_class.upper()} ({confidence:.1%})\n(Why model chose this)", fontsize=9, fontweight='bold')
                axes[row_idx, 1].axis('off')

                # 3. Ground Truth / True Class CAM (e.g. MEL or BCC evidence)
                true_cam, _ = cam_gen(img_tensor, target_class=true_label_idx)
                axes[row_idx, 2].imshow(true_cam, cmap='jet')
                axes[row_idx, 2].set_title(f"True CAM: {true_class.upper()} ({true_conf:.1%})\n(Pathology evidence)", fontsize=9, fontweight='bold')
                axes[row_idx, 2].axis('off')

                # 4. Superimposed Overlay
                axes[row_idx, 3].imshow(superimposed)
                axes[row_idx, 3].set_title(
                    f"Pred: {pred_class.upper()} ({confidence:.1%})\nResult: {'CORRECT' if is_correct else 'INCORRECT'}",
                    fontsize=10, fontweight='bold', color=status_color
                )
                axes[row_idx, 3].axis('off')
            else:
                # 2. Activation Heatmap
                title_cam = f"Grad-CAM Attention Heatmap\n(Target: {true_class.upper()})" if target_mode == "true" else "Grad-CAM CNN Attention Heatmap\n(Where the model looks)"
                axes[row_idx, 1].imshow(heatmap, cmap='jet')
                axes[row_idx, 1].set_title(title_cam, fontsize=10, fontweight='bold')
                axes[row_idx, 1].axis('off')

                # 3. Superimposed Overlay
                axes[row_idx, 2].imshow(superimposed)
                axes[row_idx, 2].set_title(
                    f"Pred: {pred_class.upper()} ({confidence:.1%})\nResult: {'CORRECT' if is_correct else 'INCORRECT'}",
                    fontsize=10, fontweight='bold', color=status_color
                )
                axes[row_idx, 2].axis('off')

        except Exception as e:
            print(f"Warning: Could not process Grad-CAM for {img_path}: {e}")

    cam_gen.remove_hooks()
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=250, bbox_inches='tight')
        plt.close(fig)

    return fig


# ─── 5. Multi-Model Benchmark Comparison ──────────────────────────────────────

def plot_benchmark_summary(
    results: Dict[str, Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """Generate a multi-model comparative bar chart across all evaluated architectures."""
    model_names = list(results.keys())
    accuracies = [results[m].get('pad_accuracy', results[m].get('accuracy', 0.0)) for m in model_names]
    f1_scores = [results[m].get('pad_mel_auc_roc', results[m].get('mel_auc_roc', 0.0)) for m in model_names]
    macro_f1s = [results[m].get('pad_macro_auc_roc', results[m].get('macro_auc_roc', 0.0)) for m in model_names]

    x = np.arange(len(model_names))
    width = 0.26

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#ffffff')

    rects1 = ax.bar(x - width, accuracies, width, label='PAD Accuracy', color='#1f77b4', edgecolor='white')
    rects2 = ax.bar(x, f1_scores, width, label='PAD Melanoma AUC', color='#ff7f0e', edgecolor='white')
    rects3 = ax.bar(x + width, macro_f1s, width, label='PAD Macro AUC', color='#2ca02c', edgecolor='white')

    ax.set_title("MobileNet V1–V5 Architecture Comparison", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Model Architecture', fontsize=11, labelpad=8)
    ax.set_ylabel('Performance Score [0.0 - 1.0]', fontsize=11, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in model_names], fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', linestyle=':', alpha=0.7)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0.01:
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=250, bbox_inches='tight')
        plt.close(fig)

    return fig


# ─── 6. In-Domain vs. Out-of-Domain Comparison ───────────────────────────────

def plot_domain_comparison(
    ham_results: Dict[str, Any],
    pad_results: Dict[str, Any],
    model_name: str = "Model",
    output_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """Generate a side-by-side comparative bar chart: In-Domain (HAM10000) vs Out-of-Domain (PAD-UFES-20)."""
    metrics_keys = ['accuracy', 'weighted_avg_f1', 'mel_recall', 'mel_auc_roc', 'macro_auc_roc', 'bcc_recall', 'akiec_recall']
    metric_labels = ['Accuracy', 'Weighted F1', 'Mel Recall', 'Mel AUC-ROC', 'Macro AUC', 'BCC Recall', 'AKIEC Recall']

    ham_scores = [ham_results.get(k, 0.0) or 0.0 for k in metrics_keys]
    pad_scores = [pad_results.get(k, 0.0) or 0.0 for k in metrics_keys]

    x = np.arange(len(metrics_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(15, 6))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#ffffff')

    rects1 = ax.bar(x - width/2, ham_scores, width, label='In-Domain: HAM10000 (Dermoscopy)', color='#1f77b4', edgecolor='white')
    rects2 = ax.bar(x + width/2, pad_scores, width, label='Out-of-Domain: PAD-UFES-20 (Smartphone)', color='#d62728', edgecolor='white')

    ax.set_title(f"{model_name.upper()} — Clinical Domain Shift: HAM10000 (Dermoscopy) vs. PAD-UFES-20 (Smartphone)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Diagnostic Score [0.0 - 1.0]', fontsize=11, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1.18)
    ax.grid(axis='y', linestyle=':', alpha=0.7)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

    for rects in (rects1, rects2):
        for rect in rects:
            height = rect.get_height()
            if height > 0.01:
                ax.annotate(f'{height:.1%}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=250, bbox_inches='tight')
        plt.close(fig)
    return fig


# ─── 7. ROC Curves & Multi-Class / Binary AUC-ROC ─────────────────────────────

def plot_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    model_name: str = "Model"
) -> plt.Figure:
    """Plot multi-class One-vs-Rest ROC curves with high-contrast distinct curves and binary Melanoma triage curve."""
    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#ffffff')

    colors = sns.color_palette("tab10", len(class_names))
    present_classes = np.unique(y_true)
    macro_aucs = []

    # Plot per-class OvR ROC curves
    for idx, name in enumerate(class_names):
        if idx in present_classes:
            y_bin = (y_true == idx).astype(int)
            if len(np.unique(y_bin)) > 1:
                fpr, tpr, _ = roc_curve(y_bin, y_probs[:, idx])
                roc_auc = auc(fpr, tpr)
                macro_aucs.append(roc_auc)
                lw = 2.8 if name.lower() == 'mel' else 1.8
                ls = '-' if name.lower() == 'mel' else '--'
                label = f"{name.upper()} (AUC = {roc_auc:.3f})"
                if name.lower() == 'mel':
                    label += " [Primary Target]"
                ax.plot(fpr, tpr, color=colors[idx % len(colors)], lw=lw, linestyle=ls, label=label)

    # Plot Macro Average
    if macro_aucs:
        mean_auc = float(np.mean(macro_aucs))
        ax.plot([], [], ' ', label=f"Macro-Average AUC = {mean_auc:.3f}")

    # Reference diagonal (random guessing)
    ax.plot([0, 1], [0, 1], color='#888888', linestyle=':', lw=1.5, label='Random Chance (AUC = 0.500)')

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title(f"{model_name.upper()} — Receiver Operating Characteristic (ROC) Curves", fontsize=12, fontweight='bold', pad=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc="lower right", frameon=True, facecolor='white', framealpha=0.95, fontsize=9.5)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=250, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_dual_roc_comparison(
    ham_results: Dict[str, Any],
    pad_results: Dict[str, Any],
    class_names: List[str],
    output_path: Optional[Union[str, Path]] = None,
    model_name: str = "Model"
) -> plt.Figure:
    """Side-by-side ROC comparison between In-Domain (HAM10000) and Out-of-Domain (PAD-UFES-20)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5))
    fig.patch.set_facecolor('#fafafa')

    colors = sns.color_palette("tab10", len(class_names))

    for ax, res, domain_title in [
        (ax1, ham_results, "In-Domain: HAM10000 (Dermoscopy)"),
        (ax2, pad_results, "Out-of-Domain: PAD-UFES-20 (Smartphone)")
    ]:
        ax.set_facecolor('#ffffff')
        y_true = res.get('all_targets')
        y_probs = res.get('all_probs')

        if y_true is not None and y_probs is not None:
            present_classes = np.unique(y_true)
            macro_aucs = []
            for idx, name in enumerate(class_names):
                if idx in present_classes:
                    y_bin = (y_true == idx).astype(int)
                    if len(np.unique(y_bin)) > 1:
                        fpr, tpr, _ = roc_curve(y_bin, y_probs[:, idx])
                        roc_auc = auc(fpr, tpr)
                        macro_aucs.append(roc_auc)
                        lw = 2.8 if name.lower() == 'mel' else 1.8
                        ls = '-' if name.lower() == 'mel' else '--'
                        label = f"{name.upper()} (AUC = {roc_auc:.3f})"
                        if name.lower() == 'mel':
                            label += " [Primary Target]"
                        ax.plot(fpr, tpr, color=colors[idx % len(colors)], lw=lw, linestyle=ls, label=label)

            if macro_aucs:
                mean_auc = float(np.mean(macro_aucs))
                ax.plot([], [], ' ', label=f"Macro-Average AUC = {mean_auc:.3f}")

        ax.plot([0, 1], [0, 1], color='#888888', linestyle=':', lw=1.5, label='Random Chance (AUC = 0.500)')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold', labelpad=8)
        ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11, fontweight='bold', labelpad=8)
        ax.set_title(f"{domain_title}\n{model_name.upper()} ROC Analysis", fontsize=11, fontweight='bold', pad=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc="lower right", frameon=True, facecolor='white', framealpha=0.95, fontsize=9)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=250, bbox_inches='tight')
        plt.close(fig)
    return fig

