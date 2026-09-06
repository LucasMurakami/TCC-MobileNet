"""
Regenerates Grad-CAM CNN Interpretability galleries across all evaluated checkpoints.
Uses the upgraded PyTorchGradCAM engine with SE-layer filtering and spatial smoothing.
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import torch
import timm

# Setup path to import workspace modules
sys.path.append(str(Path(__file__).parent))
from dataset import CLASS_NAMES, NUM_CLASSES, prepare_dataset, ensure_pad_ufes20_download, load_pad_ufes20_validation
from train_timm_models import MODEL_CONFIGS, build_transforms, configure_hardware_environment
from visualize import generate_gradcam_gallery


def parse_args():
    parser = argparse.ArgumentParser(description='Regenerate Grad-CAM heatmaps for trained checkpoints')
    parser.add_argument('--session-dir', type=str, default=None,
                        help='Path to specific experiment session directory (e.g. experiments/26_08_2026). If omitted, processes all experiment sessions.')
    parser.add_argument('--model', type=str, default=None,
                        help='Filter by specific model variant (e.g. v3, v4, v5). If omitted, processes all models.')
    parser.add_argument('--scenario', type=str, default=None,
                        help='Filter by specific scenario name (e.g. maximum, medium, low). If omitted, processes all.')
    parser.add_argument('--no-cudnn', action='store_true', default=False, help='Disable cuDNN compatibility path')
    return parser.parse_args()


def load_validation_dataframes(target_dir: Path):
    """Loads or prepares HAM10000 and PAD-UFES-20 validation dataframes."""
    # 1. HAM10000 in-domain validation set
    ham_candidates = [
        target_dir / 'ham_val_df.csv',
        target_dir.parent / 'ham_val_df.csv',
        Path('experiments/ham_val_df.csv'),
        Path('mobilenet_outputs/ham_val_df.csv'),
    ]
    ham_val_df = None
    for p in ham_candidates:
        if p.exists():
            ham_val_df = pd.read_csv(p)
            break
    if ham_val_df is None:
        print("  [dataset] Preparing HAM10000 validation split...")
        _, _, ham_val_df = prepare_dataset(Path('./data_cache'), Path('./dataset_treino'), oversample=False)

    # 2. PAD-UFES-20 out-of-domain validation set
    pad_candidates = [
        target_dir / 'pad_val_df.csv',
        target_dir.parent / 'pad_val_df.csv',
        Path('experiments/pad_val_df.csv'),
        Path('mobilenet_outputs/pad_val_df.csv'),
        Path('experiments/val_df.csv'),
    ]
    pad_val_df = None
    for p in pad_candidates:
        if p.exists():
            df = pd.read_csv(p)
            if len(df) > 1000:
                pad_val_df = df
                break
    if pad_val_df is None:
        print("  [dataset] Loading PAD-UFES-20 external validation split...")
        pad_dir = ensure_pad_ufes20_download(Path('./data_cache'))
        pad_val_df = load_pad_ufes20_validation(pad_dir)

    return ham_val_df, pad_val_df


def main():
    args = parse_args()
    hw = configure_hardware_environment(disable_cudnn=args.no_cudnn)
    device = hw['device']
    print(f"Using device: {hw['device_name']} | cuDNN: {'enabled' if hw['cudnn_enabled'] else 'disabled'}")

    # Determine target directories
    if args.session_dir:
        session_dirs = [Path(args.session_dir)]
    else:
        exp_root = Path('experiments')
        session_dirs = [d for d in exp_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
        if not session_dirs:
            session_dirs = [Path('mobilenet_outputs')]

    print(f"Found {len(session_dirs)} session directories to inspect.")

    for s_dir in sorted(session_dirs):
        print(f"\n{'='*75}")
        print(f" 📂 Inspecting Session Directory: {s_dir}")
        print(f"{'='*75}")

        # Find all best_model.pth checkpoints in this session
        checkpoint_paths = list(s_dir.glob("**/best_model.pth"))
        if not checkpoint_paths:
            print(f"  No checkpoints found in {s_dir}. Skipping.")
            continue

        ham_val_df, pad_val_df = load_validation_dataframes(s_dir)
        print(f"  Loaded validation sets: HAM10000 ({len(ham_val_df)} samples) | PAD-UFES-20 ({len(pad_val_df)} samples)")
        print(f"  Found {len(checkpoint_paths)} checkpoint(s) to process.")

        for idx, cp_path in enumerate(checkpoint_paths, start=1):
            # scenarios/<scenario>/<model>/best_model.pth  or  scenarios/<scenario>/<model>/seed<N>/best_model.pth
            model_parent = cp_path.parent
            if model_parent.name.startswith('seed') and model_parent.name[4:].isdigit():
                model_parent = model_parent.parent
            model_name = model_parent.name
            scenario_name = model_parent.parent.name

            # Canonicalize aliases
            canonical_name = model_name
            if canonical_name in ('v4conv', 'v4convl'):
                canonical_name = 'v4'
            elif canonical_name in ('v3large', 'v3small'):
                canonical_name = 'v3'

            if args.model and canonical_name != args.model and model_name != args.model:
                continue
            if args.scenario and scenario_name.lower() != args.scenario.lower():
                continue

            print(f"\n  [{idx}/{len(checkpoint_paths)}] Processing Scenario: {scenario_name.upper()} | Model: {model_name.upper()}")

            if canonical_name not in MODEL_CONFIGS and model_name not in MODEL_CONFIGS:
                print(f"    Skipping unknown model configuration: {model_name}")
                continue

            cfg = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS.get(canonical_name))
            timm_name = cfg['timm_name']
            img_size = cfg['input_size']

            print(f"    Instantiating architecture: {timm_name}...")
            model = timm.create_model(timm_name, pretrained=False, num_classes=NUM_CLASSES)

            print(f"    Loading weights from {cp_path}...")
            state_dict = torch.load(cp_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict)
            model = model.to(device)
            model.eval()
            val_transform, _ = build_transforms(model, img_size, train=False)

            # Output directories
            model_dir = cp_path.parent
            ham_dir = model_dir / 'ham10000'
            pad_dir = model_dir / 'pad_ufes_20'
            ham_dir.mkdir(parents=True, exist_ok=True)
            pad_dir.mkdir(parents=True, exist_ok=True)

            # 1. In-domain HAM10000 Grad-CAM
            print("    🖼️  Generating In-Domain (HAM10000) Grad-CAM gallery...")
            generate_gradcam_gallery(
                model=model,
                val_df=ham_val_df,
                class_names=CLASS_NAMES,
                img_size=img_size,
                output_path=ham_dir / 'gradcam_heatmaps.png',
                model_name=f"{model_name}_HAM",
                device=device,
                transform=val_transform
            )

            # 2. Out-of-domain PAD-UFES-20 Grad-CAM
            print("    🖼️  Generating Out-of-Domain (PAD-UFES-20) Grad-CAM gallery...")
            generate_gradcam_gallery(
                model=model,
                val_df=pad_val_df,
                class_names=CLASS_NAMES,
                img_size=img_size,
                output_path=pad_dir / 'gradcam_heatmaps.png',
                model_name=f"{model_name}_PAD",
                device=device,
                transform=val_transform
            )

            # 3. Top-level backward-compatible gallery
            generate_gradcam_gallery(
                model=model,
                val_df=pad_val_df,
                class_names=CLASS_NAMES,
                img_size=img_size,
                output_path=model_dir / 'gradcam_heatmaps.png',
                model_name=model_name,
                device=device,
                transform=val_transform
            )

            # Free GPU memory
            del model
            del state_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("    ✅ Done.")

    print("\n🎉 Grad-CAM heatmaps regeneration complete across all models!")


if __name__ == '__main__':
    main()
