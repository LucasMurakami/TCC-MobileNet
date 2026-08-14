"""
MobileNet V1-V4 Trainer for HAM10000
Supports: V1, V2, V3Small, V3Large, V4Conv, V4Conv-Large
Usage: python train_mobilenets.py --model v1 --epochs 50 --batch-size 32
"""

import argparse
import os
import sys
import json
import shutil
import glob
import itertools
from time import perf_counter
from pathlib import Path
from datetime import datetime

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import kagglehub
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import (
    Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, PReLU,
    GlobalAveragePooling2D, Dense, Dropout, Reshape, Multiply, Add,
    AveragePooling2D, Activation, Rescaling, Softmax
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.applications import (
    MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large
)
from sklearn.utils import resample, class_weight as sk_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ─── GPU Setup ───────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

mixed_precision.set_global_policy('mixed_float16')

# ─── Focal Loss ──────────────────────────────────────────────────────────────

class FocalLoss(tf.keras.losses.Loss):
    """Focal loss for imbalanced classification."""
    def __init__(self, gamma=2.0, name='focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        ce = -y_true * tf.math.log(y_pred)
        modulating = tf.pow(1 - y_pred, self.gamma)
        return tf.reduce_mean(modulating * ce)

    def get_config(self):
        config = super().get_config()
        config.update({'gamma': self.gamma})
        return config

# ─── Constants ───────────────────────────────────────────────────────────────
NUM_CLASSES = 7
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

MODEL_CONFIGS = {
    'v1':       {'input_size': 224, 'weights': 'imagenet', 'backbone': 'mobilenet_v1'},
    'v2':       {'input_size': 224, 'weights': 'imagenet', 'backbone': 'mobilenet_v2'},
    'v3small':  {'input_size': 224, 'weights': 'imagenet', 'backbone': 'mobilenet_v3_small'},
    'v3large':  {'input_size': 224, 'weights': 'imagenet', 'backbone': 'mobilenet_v3_large'},
    'v4conv':   {'input_size': 224, 'weights': None,      'backbone': 'mobilenet_v4_conv'},
    'v4convl':  {'input_size': 224, 'weights': None,      'backbone': 'mobilenet_v4_conv_large'},

}

PAD_UFES20_LABEL_MAP = {
    'ack': 'akiec',
    'akiec': 'akiec',
    'actinic keratosis': 'akiec',
    'bcc': 'bcc',
    'basal cell carcinoma': 'bcc',
    'mel': 'mel',
    'melanoma': 'mel',
    'nev': 'nv',
    'nevus': 'nv',
    'sek': 'bkl',
    'seborrheic keratosis': 'bkl',
}


# ─── Timing Helpers ──────────────────────────────────────────────────────────

def _format_metrics(logs):
    if not logs:
        return ''
    pieces = []
    for key, value in logs.items():
        if isinstance(value, (int, float, np.floating, np.integer)):
            pieces.append(f'{key}={float(value):.4f}')
    return ', '.join(pieces)


class EpochTimingCallback(tf.keras.callbacks.Callback):
    def __init__(self, stage_name):
        super().__init__()
        self.stage_name = stage_name
        self.stage_start = None
        self.epoch_start = None

    def on_train_begin(self, logs=None):
        self.stage_start = perf_counter()
        print(f'  [{self.stage_name}] training started')

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = perf_counter() - self.epoch_start if self.epoch_start is not None else 0.0
        metrics = _format_metrics(logs)
        epoch_total = self.params.get('epochs', epoch + 1)
        suffix = f' | {metrics}' if metrics else ''
        print(f'  [{self.stage_name}] epoch {epoch + 1}/{epoch_total} took {elapsed:.1f}s{suffix}')

    def on_train_end(self, logs=None):
        if self.stage_start is not None:
            total_elapsed = perf_counter() - self.stage_start
            print(f'  [{self.stage_name}] training finished in {total_elapsed:.1f}s')


# ─── MobileNetV4 Implementation ──────────────────────────────────────────────

def uib_block(x, filters, expand_ratio=4, kernel_size=3, stride=1, use_se=False, se_ratio=0.25, use_extra_dw=False, use_residual=True):
    """Universal Inverted Bottleneck block from MobileNetV4 paper."""
    shortcut = x
    in_channels = x.shape[-1]
    expanded_channels = in_channels * expand_ratio

    # Expansion
    if expand_ratio > 1:
        x = Conv2D(expanded_channels, 1, strides=1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

    # Depthwise
    x = DepthwiseConv2D(kernel_size, strides=stride, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Optional extra depthwise (for larger capacity)
    if use_extra_dw:
        x = DepthwiseConv2D(kernel_size, strides=1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

    # Squeeze-and-Excitation
    if use_se:
        se = GlobalAveragePooling2D()(x)
        se = Reshape((1, 1, expanded_channels))(se)
        se_channels = max(1, int(expanded_channels * se_ratio))
        se = Dense(se_channels, activation='relu')(se)
        se = Dense(expanded_channels, activation='sigmoid')(se)
        x = Multiply()([x, se])

    # Projection
    x = Conv2D(filters, 1, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Residual connection
    if use_residual and stride == 1 and in_channels == filters:
        x = Add()([shortcut, x])

    return x

def mobilenet_v4_conv(input_shape, num_classes, width_mult=1.0, variant='m'):
    """
    MobileNetV4-Conv: pure ConvNet variant using UIB blocks.
    Variants: 's' (small), 'm' (medium), 'l' (large)
    Follows the paper's architecture pattern with configurable depth/width.
    """
    variant_configs = {
        's': {
            'stem_channels': 32,
            'stages': [
                {'filters': 32,  'blocks': 2, 'expand': 2, 'stride': 2, 'use_se': False},
                {'filters': 64,  'blocks': 4, 'expand': 3, 'stride': 2, 'use_se': False},
                {'filters': 128, 'blocks': 4, 'expand': 3, 'stride': 2, 'use_se': True},
                {'filters': 256, 'blocks': 2, 'expand': 4, 'stride': 1, 'use_se': True},
            ],
            'head_channels': 1024,
        },
        'm': {
            'stem_channels': 32,
            'stages': [
                {'filters': 48,  'blocks': 2, 'expand': 3, 'stride': 2, 'use_se': False},
                {'filters': 80,  'blocks': 4, 'expand': 3, 'stride': 2, 'use_se': False},
                {'filters': 160, 'blocks': 6, 'expand': 4, 'stride': 2, 'use_se': True},
                {'filters': 320, 'blocks': 3, 'expand': 4, 'stride': 1, 'use_se': True},
            ],
            'head_channels': 1280,
        },
        'l': {
            'stem_channels': 48,
            'stages': [
                {'filters': 64,  'blocks': 3, 'expand': 3, 'stride': 2, 'use_se': False},
                {'filters': 128, 'blocks': 6, 'expand': 4, 'stride': 2, 'use_se': False},
                {'filters': 256, 'blocks': 8, 'expand': 4, 'stride': 2, 'use_se': True},
                {'filters': 512, 'blocks': 4, 'expand': 4, 'stride': 1, 'use_se': True},
            ],
            'head_channels': 1280,
        },
    }

    cfg = variant_configs.get(variant, variant_configs['m'])

    def _scale(x, factor):
        return max(8, int(x * factor))

    inputs = Input(shape=input_shape)

    x = inputs
    stem_ch = _scale(cfg['stem_channels'], width_mult)
    x = Conv2D(stem_ch, 3, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Body stages
    for stage_cfg in cfg['stages']:
        ch = _scale(stage_cfg['filters'], width_mult)
        for i in range(stage_cfg['blocks']):
            stride = stage_cfg['stride'] if i == 0 else 1
            use_extra = stage_cfg.get('use_extra_dw', False) and i == 0
            x = uib_block(
                x, filters=ch, expand_ratio=stage_cfg['expand'],
                stride=stride, use_se=stage_cfg['use_se'],
                use_extra_dw=use_extra, use_residual=True
            )

    # Head
    head_ch = _scale(cfg['head_channels'], width_mult)
    x = Conv2D(head_ch, 1, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)

    return Model(inputs, outputs)

# ─── Model Builder ───────────────────────────────────────────────────────────

def build_mobilenet_model(model_name, input_size, num_classes):
    """Build a MobileNet model for the specified version."""
    input_shape = (input_size, input_size, 3)

    if model_name == 'v1':
        base = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
        base.trainable = False
        x = GlobalAveragePooling2D()(base.output)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)
        return Model(inputs=base.input, outputs=outputs)

    elif model_name == 'v2':
        base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
        base.trainable = False
        x = GlobalAveragePooling2D()(base.output)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)
        return Model(inputs=base.input, outputs=outputs)

    elif model_name == 'v3small':
        base = MobileNetV3Small(weights='imagenet', include_top=False, input_shape=input_shape)
        base.trainable = False
        x = GlobalAveragePooling2D()(base.output)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)
        return Model(inputs=base.input, outputs=outputs)

    elif model_name == 'v3large':
        base = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=input_shape)
        base.trainable = False
        x = GlobalAveragePooling2D()(base.output)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)
        return Model(inputs=base.input, outputs=outputs)

    elif model_name == 'v4conv':
        model = mobilenet_v4_conv(input_shape, num_classes, width_mult=1.0, variant='m')
        model._name = 'mobilenet_v4_conv'
        return model

    elif model_name == 'v4convl':
        model = mobilenet_v4_conv(input_shape, num_classes, width_mult=1.0, variant='l')
        model._name = 'mobilenet_v4_conv_large'
        return model

    else:
        raise ValueError(f"Unknown model: {model_name}")

# ─── Dataset ─────────────────────────────────────────────────────────────────

def _first_existing_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


def ensure_pad_ufes20_download(cache_root: Path):
    """Download PAD-UFES-20 from Kaggle when it is not already present locally."""
    val_root = cache_root / 'pad_ufes_20_raw'
    csv_candidates = [
        val_root / 'metadata.csv',
        val_root / 'PADUFES20_metadata.csv',
        val_root / 'PAD-UFES-20_metadata.csv',
        val_root / 'PADUFES20.csv',
    ]
    if any(path.exists() for path in csv_candidates):
        return val_root

    print("[dataset] Downloading PAD-UFES-20 from Kaggle...")
    try:
        downloaded_path = Path(kagglehub.dataset_download("andrewmvd/pad-ufes-20"))
    except Exception as exc:
        raise FileNotFoundError(
            f"PAD-UFES-20 validation dataset could not be downloaded automatically. "
            f"Please download it manually and place it under {val_root}. "
            f"Original error: {exc}"
        ) from exc

    if val_root.exists():
        shutil.rmtree(val_root)
    shutil.copytree(downloaded_path, val_root)
    print("[dataset] PAD-UFES-20 download complete.")
    return val_root


def load_pad_ufes20_validation(val_root: Path):
    """Load PAD-UFES-20 as an external validation set.

    The loader accepts a local PAD-UFES-20 export with a metadata CSV and
    image files. Diagnosis labels are mapped into the HAM10000 label space
    where possible and unsupported classes are dropped.
    """
    if not val_root.exists():
        raise FileNotFoundError(
            f"PAD-UFES-20 validation root not found: {val_root}. "
            "Place the dataset there or pass --pad-ufes-dir."
        )

    csv_candidates = [
        val_root / 'metadata.csv',
        val_root / 'PADUFES20_metadata.csv',
        val_root / 'PAD-UFES-20_metadata.csv',
        val_root / 'PADUFES20.csv',
    ]
    metadata_path = next((path for path in csv_candidates if path.exists()), None)
    if metadata_path is None:
        csv_files = list(val_root.glob('*.csv')) + list(val_root.glob('**/*.csv'))
        if len(csv_files) == 1:
            metadata_path = csv_files[0]
        else:
            raise FileNotFoundError(
                f"Could not find a PAD-UFES-20 metadata CSV under {val_root}. "
                "Expected a single CSV such as metadata.csv."
            )

    df = pd.read_csv(metadata_path)
    diagnosis_col = _first_existing_column(
        df,
        ['dx', 'diagnosis', 'diagnostic', 'label', 'lesion_type']
    )
    if diagnosis_col is None:
        raise KeyError(
            f"Could not find a diagnosis column in {metadata_path}. "
            f"Available columns: {list(df.columns)}"
        )

    image_id_col = _first_existing_column(
        df,
        ['image_id', 'img_id', 'image', 'image_name', 'img_name', 'filename', 'file_name', 'name']
    )
    path_col = _first_existing_column(df, ['path', 'filepath', 'file_path'])

    if path_col is not None:
        candidate_paths = df[path_col].astype(str).map(lambda value: Path(value))
        if candidate_paths.map(lambda path: path.is_file()).all():
            df['path'] = candidate_paths.astype(str)
        else:
            df['path'] = candidate_paths.map(lambda path: path if path.is_absolute() else val_root / path)
            df['path'] = df['path'].map(str)
    elif image_id_col is not None:
        imageid_path_dict = {}
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            for file_path in glob.glob(os.path.join(str(val_root), '**', ext), recursive=True):
                imageid_path_dict[Path(file_path).stem] = file_path
        df['path'] = df[image_id_col].astype(str).map(imageid_path_dict)
    else:
        raise KeyError(
            f"Could not find an image identifier or path column in {metadata_path}. "
            f"Available columns: {list(df.columns)}"
        )

    df['dx'] = df[diagnosis_col].astype(str).str.strip().str.lower().map(PAD_UFES20_LABEL_MAP)
    unsupported_rows = int(df['dx'].isna().sum())
    df = df.dropna(subset=['dx', 'path']).copy()
    df['path'] = df['path'].astype(str)
    df = df[df['path'].map(lambda value: Path(value).is_file())].copy()

    print(
        f"[dataset] PAD-UFES-20 validation loaded from {metadata_path} "
        f"with {len(df)} samples ({unsupported_rows} unsupported rows dropped)"
    )
    return df.reset_index(drop=True)

def prepare_dataset(cache_root: Path, prepared_dir: Path):
    """Prepare HAM10000 training data and optionally an external validation set."""
    raw_dir = cache_root / "ham10000_raw"

    if not (raw_dir / "HAM10000_metadata.csv").exists():
        print("[dataset] Downloading HAM10000 from Kaggle...")
        downloaded_path = Path(kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000"))
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        shutil.copytree(downloaded_path, raw_dir)
        print("[dataset] Download complete.")

    df = pd.read_csv(raw_dir / "HAM10000_metadata.csv")

    imageid_path_dict = {}
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        for f in glob.glob(os.path.join(raw_dir, '*', ext)):
            imageid_path_dict[os.path.splitext(os.path.basename(f))[0]] = f
    df['path'] = df['image_id'].map(imageid_path_dict)

    # Use pre-split prepared_dir if it has .ready marker
    ready_file = prepared_dir / '.ready'
    if ready_file.exists():
        print(f"[dataset] Using prepared data from {prepared_dir}")
        df_local = pd.DataFrame()
        for cls in CLASS_NAMES:
            cls_dir = prepared_dir / cls
            if cls_dir.exists():
                for f in cls_dir.iterdir():
                    if f.suffix.lower() in ('.jpg', '.png', '.jpeg'):
                        df_local = pd.concat([df_local, pd.DataFrame({'path': [str(f)], 'dx': [cls]})])
        df_local = df_local.reset_index(drop=True)
        train_df, val_df = train_test_split(df_local, test_size=0.2, random_state=42, stratify=df_local['dx'])
        return train_df, val_df

    # Split first
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['dx'])

    # Oversample training
    print("[dataset] Oversampling training set...")
    max_size = train_df['dx'].value_counts().max()
    lst = []
    for _, group in train_df.groupby('dx'):
        lst.append(resample(group, replace=True, n_samples=max_size, random_state=42))
    train_df = pd.concat(lst)

    print(f"[dataset] Train: {len(train_df)}, Val: {len(val_df)}")
    return train_df, val_df


def prepare_dataset_with_external_validation(cache_root: Path, prepared_dir: Path, pad_ufes_dir: Path):
    """Prepare HAM10000 training data and PAD-UFES-20 validation data."""
    train_df, _ = prepare_dataset(cache_root, prepared_dir)
    val_dir = ensure_pad_ufes20_download(cache_root) if not pad_ufes_dir.exists() else pad_ufes_dir
    val_df = load_pad_ufes20_validation(val_dir)
    return train_df, val_df

# ─── Training ────────────────────────────────────────────────────────────────

def train_model(args, model_name, output_dir):
    """Train a single model variant and return results dict."""
    print(f"\n{'='*60}")
    print(f"  Training {model_name.upper()}")
    print(f"{'='*60}")

    cfg = MODEL_CONFIGS[model_name]
    img_size = args.img_size if args.img_size else cfg['input_size']

    # Output subdirectory
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    if args.val_dataset == 'pad-ufes-20':
        train_df, val_df = prepare_dataset_with_external_validation(
            Path(args.cache_dir),
            Path(args.prepared_dir),
            Path(args.pad_ufes_dir),
        )
    else:
        train_df, val_df = prepare_dataset(Path(args.cache_dir), Path(args.prepared_dir))

    batch_size = args.batch_size
    # Adjust batch size for smaller memory
    gpu_mem = 12227  # RTX 5070 has 12GB
    if model_name == 'v4conv' and batch_size > 32:
        batch_size = 32

    # Data augmentation (no rescaling - models handle it internally or we rescale in generator)
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=45,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        brightness_range=[0.85, 1.15],
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect',
    )
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)

    train_gen = train_datagen.flow_from_dataframe(
        train_df, x_col='path', y_col='dx', target_size=(img_size, img_size),
        batch_size=batch_size, class_mode='categorical', classes=CLASS_NAMES,
    )
    val_gen = val_datagen.flow_from_dataframe(
        val_df, x_col='path', y_col='dx', target_size=(img_size, img_size),
        batch_size=batch_size, class_mode='categorical', classes=CLASS_NAMES, shuffle=False,
    )

    # Class weights for imbalance
    class_labels_list = sorted(train_df['dx'].unique())
    class_weights_dict = sk_class_weight.compute_class_weight(
        'balanced', classes=np.array(class_labels_list), y=train_df['dx']
    )
    class_weights = {i: float(w) for i, w in enumerate(class_weights_dict)}
    print(f"[weights] {class_weights}")

    # Model
    model = build_mobilenet_model(model_name, img_size, NUM_CLASSES)
    model.summary()

    # ── Stage 1: Train head only ──
    stage1_lr = args.lr_stage1
    stage1_epochs = max(min(args.epochs // 3, 15), 1)

    model.compile(
        optimizer=Adam(stage1_lr),
        loss=FocalLoss(gamma=2.0),
        metrics=['accuracy'],
        jit_compile=False,
    )

    print(f"\n--- Stage 1: Freeze backbone, lr={stage1_lr} ---")
    stage1_start = perf_counter()
    h1 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=stage1_epochs, class_weight=class_weights,
        callbacks=[
            EpochTimingCallback('stage1'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-7),
        ],
        verbose=1,
    )
    print(f"  [stage1] fit() total time: {perf_counter() - stage1_start:.1f}s")

    # ── Stage 2: Fine-tune ──
    stage2_lr = args.lr_stage2
    checkpoint_path = str(model_dir / 'best_model.keras')

    # Unfreeze
    for layer in model.layers:
        layer.trainable = True
    # Keep BatchNorm frozen
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=Adam(stage2_lr),
        loss=FocalLoss(gamma=2.0),
        metrics=['accuracy'],
        jit_compile=False,
    )

    print(f"\n--- Stage 2: Fine-tune, lr={stage2_lr} ---")
    stage2_start = perf_counter()
    h2 = model.fit(
        train_gen, validation_data=val_gen,
        epochs=args.epochs, class_weight=class_weights,
        callbacks=[
            EpochTimingCallback('stage2'),
            ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy'),
            EarlyStopping(patience=args.patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-8),
            TensorBoard(log_dir=str(model_dir / 'logs'), histogram_freq=0),
        ],
        verbose=1,
    )
    print(f"  [stage2] fit() total time: {perf_counter() - stage2_start:.1f}s")

    # Load best weights
    if os.path.exists(checkpoint_path):
        model = tf.keras.models.load_model(checkpoint_path, custom_objects={'FocalLoss': FocalLoss})

    # ── Evaluate ──
    val_gen.reset()
    y_pred = model.predict(val_gen)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = val_gen.classes
    class_labels = list(val_gen.class_indices.keys())

    report = classification_report(
        y_true, y_pred_classes, target_names=class_labels, output_dict=True, zero_division=0
    )
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_true, y_pred_classes, target_names=class_labels, zero_division=0))

    # Save report
    with open(model_dir / 'classification_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    with open(model_dir / 'classification_report.txt', 'w') as f:
        f.write(classification_report(y_true, y_pred_classes, target_names=class_labels, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'{model_name.upper()} - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(model_dir / 'confusion_matrix.png', dpi=150)
    plt.close()

    # Training curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, (history, stage) in enumerate([(h1, 'Stage 1'), (h2, 'Stage 2')]):
        ax1, ax2 = axes[i]
        ax1.plot(history.history['accuracy'], label='Train')
        ax1.plot(history.history['val_accuracy'], label='Val')
        ax1.set_title(f'{model_name.upper()} - {stage} Accuracy')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy'); ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(history.history['loss'], label='Train')
        ax2.plot(history.history['val_loss'], label='Val')
        ax2.set_title(f'{model_name.upper()} - {stage} Loss')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss'); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(model_dir / 'training_curves.png', dpi=150)
    plt.close()

    # Summary metrics
    val_acc = max(history.history['val_accuracy'] for history in [h1, h2])
    val_loss = min(history.history['val_loss'] for history in [h1, h2])
    best_val_acc = max(report['weighted avg']['f1-score'], report['accuracy'])

    results = {
        'model': model_name,
        'img_size': img_size,
        'batch_size': batch_size,
        'stage1_lr': stage1_lr,
        'stage2_lr': stage2_lr,
        'epochs_completed': len(h1.history['loss']) + len(h2.history['loss']),
        'best_val_accuracy': float(best_val_acc),
        'weighted_avg_f1': float(report['weighted avg']['f1-score']),
        'accuracy': float(report['accuracy']),
        'per_class_f1': {k: float(v['f1-score']) for k, v in report.items() if k in class_labels},
    }

    # Save results
    with open(model_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='MobileNet V1-V4 Trainer')
    parser.add_argument('--model', type=str, default='v1',
                        choices=['v1', 'v2', 'v3small', 'v3large', 'v4conv', 'v4convl', 'all'],
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

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_to_train = [args.model] if args.model != 'all' else ['v1', 'v2', 'v3small', 'v3large', 'v4conv', 'v4convl']

    # Best params per model (heuristic based on architecture + imbalance handling)
    best_params = {
        'v1':      {'lr_stage1': 1e-3, 'lr_stage2': 2e-5, 'batch_size': 48},
        'v2':      {'lr_stage1': 1e-3, 'lr_stage2': 2e-5, 'batch_size': 40},
        'v3small': {'lr_stage1': 1e-3, 'lr_stage2': 2e-5, 'batch_size': 48},
        'v3large': {'lr_stage1': 5e-4, 'lr_stage2': 1e-5, 'batch_size': 28},
        'v4conv':  {'lr_stage1': 1e-3, 'lr_stage2': 3e-5, 'batch_size': 28},
        'v4convl': {'lr_stage1': 1e-3, 'lr_stage2': 1e-5, 'batch_size': 20},
    }

    results = {}
    for model_name in models_to_train:
        bp = best_params.get(model_name, {})
        args.lr_stage1 = bp.get('lr_stage1', args.lr_stage1)
        args.lr_stage2 = bp.get('lr_stage2', args.lr_stage2)
        args.batch_size = bp.get('batch_size', args.batch_size)

        result = train_model(args, model_name, output_dir)
        results[model_name] = result

    # Summary comparison
    if len(results) > 1:
        summary_path = output_dir / 'summary_comparison.json'
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Bar chart comparison
        models = list(results.keys())
        accs = [results[m]['accuracy'] for m in models]
        f1s = [results[m]['weighted_avg_f1'] for m in models]

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(models))
        width = 0.35
        bars1 = ax.bar(x - width/2, accs, width, label='Accuracy', color='#4ECDC4')
        bars2 = ax.bar(x + width/2, f1s, width, label='Weighted F1', color='#FF6B6B')

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Model'); ax.set_ylabel('Score')
        ax.set_title('MobileNet V1-V5 Performance Comparison')
        ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in models])
        ax.legend(); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_dir / 'comparison.png', dpi=150)
        plt.close()

        # Print comparison table
        print(f"\n{'='*70}")
        print(f"  {'Model':<12} {'Accuracy':<12} {'F1-Score':<12} {'Params(M)':<12}")
        print(f"{'='*70}")
        for m in models:
            acc = results[m]['accuracy']
            f1 = results[m]['weighted_avg_f1']
            print(f"  {m:<12} {acc:<12.4f} {f1:<12.4f}")
        print(f"{'='*70}")
        print(f"Best model: {max(results, key=lambda k: results[k]['accuracy']).upper()}")
        print(f"Results saved to: {output_dir.resolve()}")

if __name__ == '__main__':
    main()
