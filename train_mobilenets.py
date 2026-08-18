"""
Unified MobileNet Benchmark Suite (V1, V2, V3Small, V3Large, V4Conv, V4ConvL, V5)
All models loaded with official pre-trained weights from ImageNet / Hugging Face / timm.
"""

import argparse
import os
import sys
import json
import subprocess
from time import perf_counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix

from dataset import (
    NUM_CLASSES, CLASS_NAMES, HAM10000_CLASSES, PAD_UFES20_LABEL_MAP,
    prepare_dataset, prepare_dataset_with_external_validation,
    compute_class_weights
)
from visualize import (
    plot_training_curves, plot_confusion_matrices,
    plot_per_class_metrics, generate_gradcam_gallery, plot_benchmark_summary
)

# ─── Model Configurations ───────────────────────────────────────────────────
MODEL_CONFIGS = {
    'v1':       {'framework': 'timm',  'input_size': 224, 'timm_name': 'mobilenetv1_100',                        'pretrained': 'ImageNet-1k (timm)'},
    'v2':       {'framework': 'timm',  'input_size': 224, 'timm_name': 'mobilenetv2_100',                        'pretrained': 'ImageNet-1k (timm)'},
    'v3small':  {'framework': 'timm',  'input_size': 224, 'timm_name': 'mobilenetv3_small_100',                  'pretrained': 'ImageNet-1k (timm)'},
    'v3large':  {'framework': 'timm',  'input_size': 224, 'timm_name': 'mobilenetv3_large_100',                  'pretrained': 'ImageNet-1k (timm)'},
    'v4conv':   {'framework': 'timm',  'input_size': 256, 'timm_name': 'mobilenetv4_conv_medium.e500_r256_in1k', 'pretrained': 'ImageNet-1k (Hugging Face / timm)'},
    'v4convl':  {'framework': 'timm',  'input_size': 384, 'timm_name': 'mobilenetv4_conv_large.e500_r384_in1k',  'pretrained': 'ImageNet-1k (Hugging Face / timm)'},
    'v5':       {'framework': 'timm',  'input_size': 256, 'timm_name': 'mobilenetv5_300m.gemma3n',               'pretrained': 'Google Gemma3n Vision (Hugging Face / timm)'},
}


# ─── Keras Model Builder ────────────────────────────────────────────────────

def build_keras_model(model_name: str, input_size: int, num_classes: int):
    import tensorflow as tf
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.applications import MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large

    input_shape = (input_size, input_size, 3)
    if model_name == 'v1':
        base = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'v2':
        base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'v3small':
        base = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'v3large':
        base = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unknown Keras model: {model_name}")

    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)
    return Model(inputs=base.input, outputs=outputs)


# ─── Keras Trainer ──────────────────────────────────────────────────────────

def train_keras_model(args, model_name: str, output_dir: Path, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    from tensorflow.keras import mixed_precision
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

    mixed_precision.set_global_policy('mixed_float16')

    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = MODEL_CONFIGS[model_name]
    img_size = args.img_size or cfg['input_size']
    batch_size = args.batch_size

    print(f"\n{'='*70}")
    print(f" [Keras] Training {model_name.upper()} with {cfg['pretrained']}")
    print(f"{'='*70}")

    train_gen, val_gen = get_data_generators(train_df, val_df, img_size=img_size, batch_size=batch_size)
    class_weights = compute_class_weights(train_df['dx'])

    model = build_keras_model(model_name, img_size, NUM_CLASSES)
    model.summary()

    stage1_lr = args.lr_stage1
    stage1_epochs = max(min(args.epochs // 3, 15), 1)

    model.compile(optimizer=Adam(stage1_lr), loss=FocalLoss(gamma=2.0), metrics=['accuracy'], jit_compile=False)
    print(f"\n--- Stage 1: Freeze backbone, lr={stage1_lr} ---")
    h1 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=stage1_epochs, class_weight=class_weights,
        callbacks=[EpochTimingCallback('stage1'), ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-7)],
        verbose=1,
    )

    stage2_lr = args.lr_stage2
    checkpoint_path = str(model_dir / 'best_model.keras')
    for layer in model.layers:
        layer.trainable = True
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(optimizer=Adam(stage2_lr), loss=FocalLoss(gamma=2.0), metrics=['accuracy'], jit_compile=False)
    print(f"\n--- Stage 2: Fine-tune, lr={stage2_lr} ---")
    h2 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=args.epochs, class_weight=class_weights,
        callbacks=[
            EpochTimingCallback('stage2'),
            ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy'),
            EarlyStopping(patience=args.patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-8),
        ],
        verbose=1,
    )

    if os.path.exists(checkpoint_path):
        model = tf.keras.models.load_model(checkpoint_path, custom_objects={'FocalLoss': FocalLoss})

    val_gen.reset()
    y_pred = model.predict(val_gen, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = val_gen.classes
    class_labels = list(val_gen.class_indices.keys())

    report = classification_report(y_true, y_pred_classes, target_names=class_labels, output_dict=True, zero_division=0)
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_true, y_pred_classes, target_names=class_labels, zero_division=0))

    with open(model_dir / 'classification_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    plot_confusion_matrices(y_true, y_pred_classes, class_labels, output_path=model_dir / 'confusion_matrix.png', model_name=model_name)
    plot_per_class_metrics(report, class_labels, output_path=model_dir / 'per_class_metrics.png', model_name=model_name)
    plot_training_curves([h1.history, h2.history], ['Stage 1 (Head)', 'Stage 2 (Fine-Tune)'], output_path=model_dir / 'training_curves.png', model_name=model_name)

    try:
        val_sample_paths = val_df['path'].tolist()
        val_sample_labels = [class_labels.index(c) for c in val_df['dx']]
        generate_gradcam_gallery(model, val_sample_paths, val_sample_labels, class_labels, img_size=img_size, output_path=model_dir / 'gradcam_heatmaps.png', num_samples=6, model_name=model_name)
    except Exception as e:
        print(f'  Warning: Grad-CAM generation encountered {e}')

    results = {
        'model': model_name,
        'pretrained': cfg['pretrained'],
        'img_size': img_size,
        'batch_size': batch_size,
        'accuracy': float(report['accuracy']),
        'weighted_avg_f1': float(report['weighted avg']['f1-score']),
        'macro_avg_f1': float(report['macro avg']['f1-score']),
        'params': model.count_params(),
    }
    with open(model_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ─── timm Process Delegator ──────────────────────────────────────────────────

def run_timm_process(args, model_name: str, output_dir: Path, train_csv: Path, val_csv: Path) -> dict:
    cmd = [
        sys.executable, 'train_timm_models.py',
        '--model', model_name,
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--lr-stage1', str(args.lr_stage1),
        '--lr-stage2', str(args.lr_stage2),
        '--patience', str(args.patience),
        '--train-csv', str(train_csv),
        '--val-csv', str(val_csv),
        '--val-dataset', args.val_dataset,
        '--output-dir', str(output_dir),
        '--seed', str(args.seed),
    ]
    if args.img_size:
        cmd.extend(['--img-size', str(args.img_size)])

    print(f"\n[Subprocess] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)

    results_file = output_dir / model_name / 'results.json'
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return {'model': model_name, 'accuracy': 0.0, 'weighted_avg_f1': 0.0}


# ─── Unified Model Router ───────────────────────────────────────────────────

def train_model(args, model_name: str, output_dir: Path) -> dict:
    if args.val_dataset == 'pad-ufes-20':
        train_df, val_df = prepare_dataset_with_external_validation(
            Path(args.cache_dir), Path(args.prepared_dir), Path(args.pad_ufes_dir),
        )
    else:
        train_df, val_df = prepare_dataset(Path(args.cache_dir), Path(args.prepared_dir))

    train_csv_path = output_dir / 'train_df.csv'
    val_csv_path = output_dir / 'val_df.csv'
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)

    cfg = MODEL_CONFIGS[model_name]
    if cfg['framework'] == 'timm':
        return run_timm_process(args, model_name, output_dir, train_csv_path, val_csv_path)
    else:
        return train_keras_model(args, model_name, output_dir, train_df, val_df)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Unified MobileNet V1-V5 Benchmark Suite')
    parser.add_argument('--model', type=str, default='v1',
                        choices=['v1', 'v2', 'v3', 'v3small', 'v3large', 'v4', 'v4conv', 'v4convl', 'v5', 'all'],
                        help='Model variant to train')
    parser.add_argument('--epochs', type=int, default=50, help='Max epochs for stage 2')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr-stage1', type=float, default=1e-3, help='Learning rate for stage 1 (head only)')
    parser.add_argument('--lr-stage2', type=float, default=1e-4, help='Learning rate for stage 2 (fine-tune)')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--img-size', type=int, default=None, help='Input image size (default: model default)')
    parser.add_argument('--cache-dir', type=str, default='./data_cache', help='Dataset cache directory')
    parser.add_argument('--prepared-dir', type=str, default='./dataset_treino', help='Prepared dataset directory')
    parser.add_argument('--val-dataset', type=str, default='ham10000', choices=['ham10000', 'pad-ufes-20'], help='Validation dataset source')
    parser.add_argument('--pad-ufes-dir', type=str, default='./data_cache/pad_ufes_20_raw', help='PAD-UFES-20 dataset directory')
    parser.add_argument('--output-dir', type=str, default='./mobilenet_outputs', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Aliases
    if args.model == 'v4':
        args.model = 'v4conv'
    elif args.model == 'v3':
        args.model = 'v3large'

    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_to_train = [args.model] if args.model != 'all' else ['v1', 'v2', 'v3small', 'v3large', 'v4conv', 'v4convl', 'v5']

    results = {}
    for model_name in models_to_train:
        result = train_model(args, model_name, output_dir)
        results[model_name] = result

    # Summary benchmark comparison
    if len(results) > 1:
        summary_path = output_dir / 'summary_comparison.json'
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)

        plot_benchmark_summary(results, output_dir / 'benchmark_comparison.png')

        print(f"\n{'='*75}")
        print(f"  {'Model':<10} {'Pretrained Source':<32} {'Accuracy':<10} {'F1-Score':<10}")
        print(f"{'='*75}")
        for m in results:
            acc = results[m]['accuracy']
            f1 = results[m]['weighted_avg_f1']
            src = results[m].get('pretrained', '')[:30]
            print(f"  {m:<10} {src:<32} {acc:<10.4f} {f1:<10.4f}")
        print(f"{'='*75}")
        best_m = max(results, key=lambda k: results[k]['accuracy'])
        print(f"🏆 Best Performing Model: {best_m.upper()} (Acc: {results[best_m]['accuracy']:.2%})")


if __name__ == '__main__':
    main()
