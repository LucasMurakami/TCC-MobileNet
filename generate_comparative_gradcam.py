import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from train_timm_models import build_transforms
from visualize import PyTorchGradCAM, overlay_gradcam, generate_gradcam_gallery

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
NUM_CLASSES = 7

MODEL_CONFIGS = {
    'v1': {'timm_name': 'mobilenetv1_100', 'input_size': 224, 'display_name': 'MobileNet V1'},
    'v2': {'timm_name': 'mobilenetv2_100', 'input_size': 224, 'display_name': 'MobileNet V2'},
    'v3': {'timm_name': 'mobilenetv3_large_100', 'input_size': 224, 'display_name': 'MobileNet V3 Large'},
    'v4': {'timm_name': 'mobilenetv4_conv_medium.e500_r256_in1k', 'input_size': 256, 'display_name': 'MobileNet V4 Conv'},
    'v5': {'timm_name': 'mobilenetv5_300m.gemma3n', 'input_size': 256, 'display_name': 'MobileNet V5 (Pre-Norm Grad-CAM)'},
}

import argparse

parser = argparse.ArgumentParser(description='Generate Comparative Grad-CAM / LayerCAM Visualizations')
parser.add_argument('--session-dir', type=str, default='experiments/01_09_2026' if Path('experiments/01_09_2026').exists() else 'experiments/30_08_2026')
parser.add_argument('--scenario', type=str, default='standard')
parser.add_argument('--target-mode', type=str, default='pred', choices=['pred', 'true', 'contrastive', 'mel', 'bcc'],
                    help='Class targeting mode for Grad-CAM (pred, true, contrastive, mel, bcc)')
cli_args, _ = parser.parse_known_args()

session_dir = Path(cli_args.session_dir)
scenario = cli_args.scenario
target_mode = cli_args.target_mode
models = ['v1', 'v2', 'v3', 'v4', 'v5']

ham_df = pd.read_csv(session_dir / 'ham_val_df.csv')
pad_df = pd.read_csv(session_dir / 'pad_val_df.csv')

def prepare_samples(df):
    samples = []
    for cls in CLASS_NAMES:
        sub = df[df['dx'] == cls]
        if len(sub) > 0:
            p = sub.iloc[0]['path']
            pil_img = Image.open(p).convert('RGB')
            img_np = np.array(pil_img.resize((256, 256)))
            samples.append({
                'true_cls': cls,
                'img_path': p,
                'pil_img': pil_img,
                'img_np': img_np,
                'models': {}
            })
    return samples

ham_samples = prepare_samples(ham_df)
pad_samples = prepare_samples(pad_df)

for m_name in models:
    cp_path = session_dir / 'scenarios' / scenario / m_name / 'best_model.pth'
    if not cp_path.exists():
        print(f"Skipping {m_name} (no checkpoint)", flush=True)
        continue

    cfg = MODEL_CONFIGS[m_name]
    print(f"Processing model: {m_name} ({cfg['display_name']})...", flush=True)
    m = timm.create_model(cfg['timm_name'], pretrained=False, num_classes=NUM_CLASSES)
    m.load_state_dict(torch.load(cp_path, map_location=device, weights_only=True))
    m = m.to(device)
    m.eval()

    cam_engine = PyTorchGradCAM(m)
    img_size = cfg['input_size']
    transform, _ = build_transforms(m, img_size, train=False)

    for s_list in [ham_samples, pad_samples]:
        for s in s_list:
            t_in = transform(s['pil_img']).unsqueeze(0).to(device)
            target_cls_idx = None
            if target_mode == 'true':
                target_cls_idx = CLASS_NAMES.index(s['true_cls'])
            elif target_mode in CLASS_NAMES:
                target_cls_idx = CLASS_NAMES.index(target_mode)

            heatmap, probs = cam_engine(t_in, target_class=target_cls_idx)
            pred_idx = int(np.argmax(probs))
            pred_cls = CLASS_NAMES[pred_idx]
            conf = float(probs[pred_idx])
            overlay = overlay_gradcam(s['img_np'], heatmap, alpha=0.48, colormap='jet')
            s['models'][m_name] = {
                'overlay': overlay,
                'pred_cls': pred_cls,
                'conf': conf,
                'title': cfg['display_name']
            }

    # If V5, also regenerate its individual model galleries
    if m_name == 'v5':
        v5_dir = session_dir / 'scenarios' / scenario / 'v5'
        print("Regenerating V5 individual galleries...", flush=True)
        gallery_mode = target_mode if target_mode in ('pred', 'true', 'contrastive') else 'pred'
        generate_gradcam_gallery(
            model=m,
            val_df=ham_df,
            class_names=CLASS_NAMES,
            img_size=img_size,
            output_path=v5_dir / 'ham10000' / 'gradcam_heatmaps.png',
            model_name='v5',
            device=device,
            target_mode=gallery_mode,
            transform=transform
        )
        generate_gradcam_gallery(
            model=m,
            val_df=pad_df,
            class_names=CLASS_NAMES,
            img_size=img_size,
            output_path=v5_dir / 'pad_ufes_20' / 'gradcam_heatmaps.png',
            model_name='v5',
            device=device,
            target_mode=gallery_mode,
            transform=transform
        )
        generate_gradcam_gallery(
            model=m,
            val_df=ham_df,
            class_names=CLASS_NAMES,
            img_size=img_size,
            output_path=v5_dir / 'gradcam_heatmaps.png',
            model_name='v5',
            device=device,
            target_mode=gallery_mode,
            transform=transform
        )

    cam_engine.remove_hooks()
    del m
    del cam_engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def plot_grid(samples, out_path, title):
    num_rows = len(samples)
    num_cols = 1 + len(models)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3.3 * num_cols, 3.4 * num_rows))
    fig.patch.set_facecolor('#ffffff')

    for row_idx, s in enumerate(samples):
        true_cls = s['true_cls']
        img_p = s['img_path']

        axes[row_idx, 0].imshow(s['img_np'])
        axes[row_idx, 0].set_title(f"Target: {true_cls.upper()}\n({Path(img_p).name})", fontsize=10, fontweight='bold')
        axes[row_idx, 0].axis('off')

        for col_idx, m_name in enumerate(models, start=1):
            if m_name not in s['models']:
                axes[row_idx, col_idx].axis('off')
                continue

            mo = s['models'][m_name]
            pred_cls = mo['pred_cls']
            conf = mo['conf']
            is_corr = (true_cls == pred_cls)
            res_color = '#1b5e20' if is_corr else '#b71c1c'

            axes[row_idx, col_idx].imshow(mo['overlay'])
            axes[row_idx, col_idx].set_title(
                f"{mo['title']}\nPred: {pred_cls.upper()} ({conf:.1%})",
                fontsize=9,
                fontweight='bold',
                color=res_color
            )
            axes[row_idx, col_idx].axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}", flush=True)

plot_grid(
    ham_samples,
    session_dir / 'gradcam_multimodel_comparison_ham10000.png',
    "Cross-Generational Interpretability Benchmark: MobileNet (V1-V5) on HAM10000"
)
plot_grid(
    ham_samples,
    session_dir / 'gradcam_multimodel_comparison.png',
    "MobileNet Generational Evolution (V1 to V5): Interpretability & Localization"
)
plot_grid(
    pad_samples,
    session_dir / 'gradcam_multimodel_comparison_pad_ufes.png',
    "Cross-Generational Interpretability Benchmark: MobileNet (V1-V5) on PAD-UFES-20"
)

print("All comparative multi-model and V5 Grad-CAM galleries successfully updated!", flush=True)
