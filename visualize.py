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
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
from torchvision import transforms

torch.backends.cudnn.enabled = False


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


# ─── 4. PyTorch Grad-CAM (CNN Heatmaps) ───────────────────────────────────────

class PyTorchGradCAM:
    """Computes Grad-CAM activation heatmaps for any PyTorch/timm CNN backbone."""
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_handles = []

        raw_model = model.module if hasattr(model, 'module') else model

        if self.target_layer is None:
            # Find the deepest convolutional layer in the architecture
            for name, module in reversed(list(raw_model.named_modules())):
                if isinstance(module, nn.Conv2d):
                    self.target_layer = module
                    break

        if self.target_layer is not None:
            self.hook_handles.append(self.target_layer.register_forward_hook(self._forward_hook))
            self.hook_handles.append(self.target_layer.register_full_backward_hook(self._backward_hook))

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, target_class: Optional[int] = None) -> tuple:
        self.model.eval()
        self.model.zero_grad()
        out = self.model(x)

        if target_class is None:
            target_class = out.argmax(dim=1).item()

        score = out[0, target_class]
        score.backward()

        if self.activations is None or self.gradients is None:
            return np.zeros((x.shape[2], x.shape[3])), out.softmax(dim=1)[0].detach().cpu().numpy()

        # Global average pool gradients over spatial dimensions
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.relu(torch.sum(weights * self.activations, dim=1, keepdim=True))

        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = nn.functional.interpolate(cam, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        heatmap_np = cam[0, 0].cpu().numpy()
        probs_np = out.softmax(dim=1)[0].detach().cpu().numpy()
        return heatmap_np, probs_np

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()


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
) -> Optional[plt.Figure]:
    """
    Generates a CNN interpretability Grad-CAM gallery containing at least ONE representative
    sample for EVERY diagnostic class present in the validation dataset.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    raw_model = model.module if hasattr(model, 'module') else model
    cam_gen = PyTorchGradCAM(raw_model)

    # Find at least 1 sample per class
    samples = []
    for cls in class_names:
        matching = val_df[val_df['dx'] == cls]
        if len(matching) > 0:
            samples.append((cls, matching.iloc[0]['path']))

    if not samples:
        cam_gen.remove_hooks()
        return None

    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 3, figsize=(13, 3.8 * num_samples))
    fig.patch.set_facecolor('#fafafa')

    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

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
            heatmap, probs = cam_gen(img_tensor, target_class=None)

            pred_label_idx = int(np.argmax(probs))
            pred_class = class_names[pred_label_idx]
            confidence = float(probs[pred_label_idx])
            superimposed = overlay_gradcam(img_np, heatmap, alpha=0.45, colormap='jet')

            is_correct = (true_label_idx == pred_label_idx)
            status_color = '#2ca02c' if is_correct else '#d62728'

            # 1. Original Image
            axes[row_idx, 0].imshow(img_np)
            axes[row_idx, 0].set_title(f"Input: {Path(img_path).name}\nTrue Class: {true_class.upper()}", fontsize=10, fontweight='bold')
            axes[row_idx, 0].axis('off')

            # 2. Grad-CAM Activation Heatmap
            axes[row_idx, 1].imshow(heatmap, cmap='jet')
            axes[row_idx, 1].set_title("Grad-CAM CNN Attention Heatmap\n(Where the model looks)", fontsize=10, fontweight='bold')
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
    accuracies = [results[m].get('accuracy', 0.0) for m in model_names]
    f1_scores = [results[m].get('weighted_avg_f1', 0.0) for m in model_names]
    macro_f1s = [results[m].get('macro_avg_f1', 0.0) for m in model_names]

    x = np.arange(len(model_names))
    width = 0.26

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#ffffff')

    rects1 = ax.bar(x - width, accuracies, width, label='Accuracy', color='#1f77b4', edgecolor='white')
    rects2 = ax.bar(x, f1_scores, width, label='Weighted F1', color='#ff7f0e', edgecolor='white')
    rects3 = ax.bar(x + width, macro_f1s, width, label='Macro F1', color='#2ca02c', edgecolor='white')

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
