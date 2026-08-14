"""
MobileNetV5 Standalone Trainer for HAM10000
Pure TF/Keras implementation (no keras_hub dependency)
Architecture: ConvNeXt-style blocks + RMSNorm + Multi-Scale Feature Aggregation
"""
import argparse
import os
import sys
import json
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

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import (
    Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Dropout,
    Add, Activation, Rescaling, AveragePooling2D, Reshape
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.regularizers import l2
from sklearn.utils import class_weight as sk_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

mixed_precision.set_global_policy('mixed_float16')

# ─── V5 Building Blocks ──────────────────────────────────────────────────────

class SafeRMSNorm(tf.keras.layers.Layer):
    def __init__(self, eps=1e-6, gamma_initializer='ones', **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.gamma_initializer = gamma_initializer

    def build(self, input_shape):
        self.dim = input_shape[-1]
        self.gamma = self.add_weight(
            shape=(self.dim,), initializer=self.gamma_initializer,
            trainable=True, name='gamma',
        )

    def call(self, x):
        x_f32 = tf.cast(x, tf.float32)
        ms = tf.reduce_mean(tf.square(x_f32), axis=-1, keepdims=True)
        norm = x_f32 * tf.math.rsqrt(ms + self.eps)
        return tf.cast(norm, x.dtype) * self.gamma


class ConvNormAct(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=1, stride=1, padding='same',
                 groups=1, bias=False, act='gelu', **kwargs):
        super().__init__(**kwargs)
        self.conv = Conv2D(filters, kernel_size, strides=stride,
                           padding=padding, groups=groups, use_bias=bias,
                           kernel_initializer='he_normal')
        self.norm = SafeRMSNorm()
        self.act = tf.keras.layers.Activation(lambda x: tf.keras.activations.gelu(x, approximate=False)) if act == 'gelu' else Activation(act)

    def call(self, x):
        return self.act(self.norm(self.conv(x)))


class UniversalInvertedResidual(tf.keras.layers.Layer):
    def __init__(self, dim, exp_ratio=4, dw_kernel=7, stride=1,
                 layer_scale=1e-5, drop_path=0.0, noskip=False, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.stride = stride
        self.drop_path_rate = drop_path
        self.layer_scale_init = layer_scale
        hidden_dim = int(dim * exp_ratio)

        self.use_skip = stride == 1 and not noskip

        self.dw_conv = DepthwiseConv2D(dw_kernel, strides=stride,
                                       padding='same', depthwise_initializer='he_normal')
        self.dw_norm = SafeRMSNorm()
        self.dw_act = tf.keras.layers.Activation(lambda x: tf.keras.activations.gelu(x, approximate=False))

        self.pw1 = Conv2D(hidden_dim, 1, use_bias=True, kernel_initializer='he_normal')
        self.pw1_act = tf.keras.layers.Activation(lambda x: tf.keras.activations.gelu(x, approximate=False))

        self.pw2 = Conv2D(dim, 1, use_bias=True, kernel_initializer='he_normal')

        if self.layer_scale_init is not None:
            self.gamma = self.add_weight(
                shape=(dim,), initializer=tf.keras.initializers.Constant(layer_scale),
                trainable=True, name='layer_scale'
            )

    def call(self, x, training=None):
        shortcut = x
        x = self.dw_conv(x)
        x = self.dw_norm(x)
        x = self.dw_act(x)
        x = self.pw1(x)
        x = self.pw1_act(x)
        x = self.pw2(x)
        if hasattr(self, 'gamma'):
            x = x * self.gamma
        if self.use_skip and shortcut.shape[-1] == x.shape[-1]:
            if self.drop_path_rate > 0 and training:
                keep_prob = 1.0 - self.drop_path_rate
                mask = tf.random.uniform(tf.shape(x)[:-1], dtype=x.dtype)[..., tf.newaxis]
                x = x / keep_prob * tf.cast(mask > self.drop_path_rate, x.dtype)
            x = tf.cast(shortcut, x.dtype) + x
        return x


class MultiScaleFusion(tf.keras.layers.Layer):
    def __init__(self, out_dim, exp_ratio=2, output_res=16, **kwargs):
        super().__init__(**kwargs)
        self.out_dim = out_dim
        self.output_res = output_res
        self.ffn = UniversalInvertedResidual(out_dim, exp_ratio=exp_ratio, dw_kernel=1, stride=1, noskip=True)
        self.norm = SafeRMSNorm()

    def call(self, features):
        resized = []
        target_h = self.output_res
        target_w = self.output_res
        for f in features:
            if f.shape[1] != target_h or f.shape[2] != target_w:
                f = tf.image.resize(f, [target_h, target_w], method='nearest')
            resized.append(f)
        x = tf.concat(resized, axis=-1)
        x = self.ffn(x)
        x = self.norm(x)
        return x


# ─── Architecture Configs ────────────────────────────────────────────────────

V5_CONFIGS = {
    'b0': {  # ~5M params, good for HAM10000
        'stem_filters': 32,
        'stages': [
            {'filters': 64,  'blocks': 2, 'stride': 2, 'exp_ratio': 4, 'dw_kernel': 7},
            {'filters': 128, 'blocks': 3, 'stride': 2, 'exp_ratio': 4, 'dw_kernel': 5},
            {'filters': 256, 'blocks': 4, 'stride': 2, 'exp_ratio': 4, 'dw_kernel': 5},
            {'filters': 512, 'blocks': 2, 'stride': 2, 'exp_ratio': 4, 'dw_kernel': 3},
        ],
        'msfa_out': 1024,
        'msfa_res': 16,
        'classifier_units': 512,
        'dropout': 0.3,
    },
    'b1': {  # ~12M params
        'stem_filters': 48,
        'stages': [
            {'filters': 96,  'blocks': 3, 'stride': 2, 'exp_ratio': 4, 'dw_kernel': 7},
            {'filters': 192, 'blocks': 4, 'stride': 2, 'exp_ratio': 4, 'dw_kernel': 5},
            {'filters': 384, 'blocks': 6, 'stride': 2, 'exp_ratio': 4, 'dw_kernel': 5},
            {'filters': 768, 'blocks': 3, 'stride': 2, 'exp_ratio': 4, 'dw_kernel': 3},
        ],
        'msfa_out': 1536,
        'msfa_res': 16,
        'classifier_units': 768,
        'dropout': 0.3,
    },
}


# ─── Model Builder ────────────────────────────────────────────────────────────

def build_mobilenetv5(input_shape=(224, 224, 3), num_classes=7, config='b0'):
    cfg = V5_CONFIGS[config]

    inputs = Input(shape=input_shape)
    x = Rescaling(1./255)(inputs)

    x = ConvNormAct(cfg['stem_filters'], kernel_size=3, stride=2)(x)

    stage_outputs = []
    for stage_cfg in cfg['stages']:
        for i in range(stage_cfg['blocks']):
            stride = stage_cfg['stride'] if i == 0 else 1
            x = UniversalInvertedResidual(
                dim=stage_cfg['filters'], exp_ratio=stage_cfg['exp_ratio'],
                dw_kernel=stage_cfg['dw_kernel'], stride=stride,
                name=f'stage_{stage_cfg["filters"]}_block_{i}'
            )(x)
        stage_outputs.append(x)

    msfa = MultiScaleFusion(out_dim=cfg['msfa_out'], output_res=cfg['msfa_res'])
    x = msfa(stage_outputs[-2:])

    x = GlobalAveragePooling2D()(x)
    x = Dropout(cfg['dropout'])(x)
    x = Dense(cfg['classifier_units'], activation='gelu',
              kernel_initializer='he_normal',
              kernel_regularizer=l2(1e-4))(x)
    x = Dropout(cfg['dropout'])(x)
    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = Model(inputs=inputs, outputs=outputs, name=f'mobilenetv5_{config}')
    return model


# ─── Dataset ──────────────────────────────────────────────────────────────────

HAM10000_CLASSES = {
    'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3,
    'mel': 4, 'nv': 5, 'vasc': 6,
}
CLASS_NAMES = list(HAM10000_CLASSES.keys())

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


def _first_existing_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


def load_pad_ufes20_validation(data_dir, img_size=224):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f'PAD-UFES-20 validation root not found: {data_dir}. '
            'Place the dataset there or pass --pad-ufes-dir.'
        )

    csv_candidates = [
        data_dir / 'metadata.csv',
        data_dir / 'PADUFES20_metadata.csv',
        data_dir / 'PAD-UFES-20_metadata.csv',
        data_dir / 'PADUFES20.csv',
    ]
    metadata_path = next((path for path in csv_candidates if path.exists()), None)
    if metadata_path is None:
        csv_files = list(data_dir.glob('*.csv')) + list(data_dir.glob('**/*.csv'))
        if len(csv_files) == 1:
            metadata_path = csv_files[0]
        else:
            raise FileNotFoundError(
                f'Could not find a PAD-UFES-20 metadata CSV under {data_dir}. '
                'Expected a single CSV such as metadata.csv.'
            )

    df = pd.read_csv(metadata_path)
    diagnosis_col = _first_existing_column(
        df,
        ['dx', 'diagnosis', 'diagnostic', 'label', 'lesion_type']
    )
    if diagnosis_col is None:
        raise KeyError(
            f'Could not find a diagnosis column in {metadata_path}. '
            f'Available columns: {list(df.columns)}'
        )

    image_id_col = _first_existing_column(
        df,
        ['image_id', 'img_id', 'image', 'image_name', 'img_name', 'filename', 'file_name', 'name']
    )
    path_col = _first_existing_column(df, ['path', 'filepath', 'file_path'])

    if path_col is not None:
        candidate_paths = df[path_col].astype(str).map(Path)
        if candidate_paths.map(lambda path: path.is_file()).all():
            df['path'] = candidate_paths.astype(str)
        else:
            df['path'] = candidate_paths.map(lambda path: path if path.is_absolute() else data_dir / path)
            df['path'] = df['path'].map(str)
    elif image_id_col is not None:
        imageid_path_dict = {}
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            for file_path in data_dir.glob(f'**/{ext}'):
                imageid_path_dict[file_path.stem] = str(file_path)
        df['path'] = df[image_id_col].astype(str).map(imageid_path_dict)
    else:
        raise KeyError(
            f'Could not find an image identifier or path column in {metadata_path}. '
            f'Available columns: {list(df.columns)}'
        )

    df['dx'] = df[diagnosis_col].astype(str).str.strip().str.lower().map(PAD_UFES20_LABEL_MAP)
    unsupported_rows = int(df['dx'].isna().sum())
    df = df.dropna(subset=['dx', 'path']).copy()
    df['path'] = df['path'].astype(str)
    df = df[df['path'].map(lambda value: Path(value).is_file())].copy()

    print(
        f'[dataset] PAD-UFES-20 validation loaded from {metadata_path} '
        f'with {len(df)} samples ({unsupported_rows} unsupported rows dropped)'
    )
    return df.reset_index(drop=True)

def load_dataset(data_dir, img_size=224, val_split=0.15, test_split=0.15,
                 oversample=True, seed=42):
    data_dir = Path(data_dir)
    images, labels = [], []
    for cls_name, cls_idx in HAM10000_CLASSES.items():
        cls_dir = data_dir / cls_name
        if not cls_dir.exists():
            print(f'  Warning: {cls_dir} not found')
            continue
        for f in cls_dir.iterdir():
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                images.append(str(f))
                labels.append(cls_idx)

    images = np.array(images)
    labels = np.array(labels)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        images, labels, test_size=test_split, stratify=labels, random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, y_train_val, test_size=val_split / (1 - test_split),
        stratify=y_train_val, random_state=seed)

    if oversample:
        df = pd.DataFrame({'path': x_train, 'label': y_train})
        majority = df['label'].value_counts().idxmax()
        majority_count = df['label'].value_counts().max()
        groups = []
        for cls in sorted(df['label'].unique()):
            grp = df[df['label'] == cls]
            if cls == majority:
                groups.append(grp)
            else:
                n_repeat = int(np.ceil(majority_count / len(grp)))
                groups.append(pd.concat([grp] * n_repeat, ignore_index=True).sample(
                    majority_count, replace=False, random_state=seed))
        df_os = pd.concat(groups, ignore_index=True)
        x_train, y_train = df_os['path'].values, df_os['label'].values
        print(f'  Oversampled: {len(y_train)} ({majority_count} per class)')

    print(f'  Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}')
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def load_dataset_with_external_validation(data_dir, val_data_dir, img_size=224, test_split=0.15,
                                          oversample=True, seed=42):
    (x_train, y_train), _, (x_test, y_test) = load_dataset(
        data_dir, img_size=img_size, val_split=0.15, test_split=test_split,
        oversample=oversample, seed=seed,
    )
    val_df = load_pad_ufes20_validation(val_data_dir, img_size=img_size)
    x_val = val_df['path'].values
    y_val = val_df['dx'].map(HAM10000_CLASSES).values
    if pd.isna(y_val).any():
        raise ValueError('PAD-UFES-20 validation contains labels outside the HAM10000 mapping.')
    return (x_train, y_train), (x_val, y_val.astype(int)), (x_test, y_test)


def make_tf_dataset(paths, labels, batch_size, img_size=224, augment=False, shuffle=True):
    def parse(p, l):
        img = tf.io.read_file(p)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32)
        return img, l

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(5000, seed=42)
    ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ])
        ds = ds.map(lambda x, y: (aug(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ─── Focal Loss ──────────────────────────────────────────────────────────────

def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -tf.math.log(tf.gather(y_pred, tf.cast(y_true, tf.int32), batch_dims=1))
        pt = tf.gather(y_pred, tf.cast(y_true, tf.int32), batch_dims=1)
        return tf.reduce_mean(tf.pow(1.0 - pt, gamma) * ce)
    return loss


# ─── Training ────────────────────────────────────────────────────────────────


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

def train_stage(model, train_ds, val_ds, test_paths, test_labels,
                stage_name, lr, epochs, output_dir):
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=focal_loss(gamma=2.0),
        metrics=['accuracy'],
        jit_compile=False,
    )

    callbacks = [
        ModelCheckpoint(str(output_dir / f'{stage_name}_best.weights.h5'),
                        monitor='val_accuracy', save_best_only=True,
                        save_weights_only=True, verbose=0),
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True,
                      verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7,
                          verbose=0),
        EpochTimingCallback(stage_name),
        TensorBoard(str(output_dir / 'logs' / stage_name), write_graph=False),
    ]

    fit_start = perf_counter()
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs, callbacks=callbacks,
        verbose=0,
    )
    fit_elapsed = perf_counter() - fit_start
    print(f'  [{stage_name}] fit() total time: {fit_elapsed:.1f}s')

    eval_start = perf_counter()
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)[:2]
    eval_elapsed = perf_counter() - eval_start
    print(f'  [{stage_name}] validation evaluation took {eval_elapsed:.1f}s')
    return history, val_acc


def evaluate(model, test_paths, test_labels, class_names, output_dir, tag=''):
    test_ds = make_tf_dataset(test_paths, test_labels, batch_size=32,
                              augment=False, shuffle=False)
    y_pred = []
    for batch_x, _ in test_ds:
        y_pred.append(model(batch_x, training=False))
    y_pred = np.concatenate(y_pred)
    y_pred_classes = np.argmax(y_pred, axis=1)

    report = classification_report(test_labels, y_pred_classes,
                                   target_names=class_names, output_dict=True,
                                   zero_division=0)
    cm = confusion_matrix(test_labels, y_pred_classes)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names, ax=ax[0])
    ax[0].set_title(f'{tag} Confusion Matrix')
    ax[0].set_ylabel('True')
    ax[0].set_xlabel('Predicted')

    classes = list(report.keys())
    metrics_df = pd.DataFrame({
        c: {'precision': report[c]['precision'], 'recall': report[c]['recall'],
            'f1-score': report[c]['f1-score'], 'support': report[c]['support']}
        for c in classes if c not in ('accuracy', 'macro avg', 'weighted avg')
    }).T
    metrics_df.plot(kind='bar', ax=ax[1])
    ax[1].set_title(f'{tag} Per-Class Metrics')
    ax[1].set_ylabel('Score')
    ax[1].set_ylim(0, 1)
    ax[1].legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(str(output_dir / f'{tag}_evaluation.png'), dpi=150)
    plt.close()

    return report, cm


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='MobileNetV5 Trainer')
    parser.add_argument('--config', default='b0', choices=['b0', 'b1'],
                        help='Model variant (b0=~5M, b1=~12M)')
    parser.add_argument('--epochs-stage1', type=int, default=20)
    parser.add_argument('--epochs-stage2', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr-stage1', type=float, default=1e-3)
    parser.add_argument('--lr-stage2', type=float, default=1e-4)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--data-dir', default='dataset_treino')
    parser.add_argument('--val-dataset', default='ham10000', choices=['ham10000', 'pad-ufes-20'])
    parser.add_argument('--pad-ufes-dir', default='./data_cache/pad_ufes_20_raw')
    parser.add_argument('--output-dir', default='mobilenetv5_output')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f'Using GPU: {gpus[0].name}')
    else:
        print('No GPU found, training on CPU')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n=== MobileNetV5-{args.config.upper()} Trainer ===')
    print(f'Output: {output_dir.resolve()}')

    print('\nLoading dataset...')
    if args.val_dataset == 'pad-ufes-20':
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_dataset_with_external_validation(
            args.data_dir, args.pad_ufes_dir, img_size=args.img_size,
            test_split=0.15, oversample=True, seed=args.seed,
        )
    else:
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_dataset(
            args.data_dir, img_size=args.img_size,
            val_split=0.15, test_split=0.15,
            oversample=True, seed=args.seed,
        )

    classes = np.unique(y_train)
    class_weights = sk_class_weight.compute_class_weight(
        'balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    print(f'Class weights: {class_weight_dict}')

    train_ds = make_tf_dataset(x_train, y_train, args.batch_size, args.img_size, augment=True)
    val_ds = make_tf_dataset(x_val, y_val, args.batch_size, args.img_size, augment=False)

    print(f'\nBuilding MobileNetV5-{args.config.upper()}...')
    model = build_mobilenetv5(input_shape=(args.img_size, args.img_size, 3),
                              num_classes=7, config=args.config)
    total = int(sum(np.prod(w.shape.as_list()) for w in model.weights))
    trainable = int(sum(np.prod(w.shape.as_list()) for w in model.trainable_weights))
    print(f'  Total params: {total:,}')
    print(f'  Trainable:    {trainable:,}')

    model.summary()

    # Stage 1: full training from scratch
    print(f'\nStage 1: Training (lr={args.lr_stage1}, {args.epochs_stage1} epochs)')
    hist1, val_acc1 = train_stage(
        model, train_ds, val_ds, x_test, y_test,
        'stage1', args.lr_stage1, args.epochs_stage1, output_dir,
    )
    report1, cm1 = evaluate(model, x_test, y_test, CLASS_NAMES, output_dir, tag='stage1')
    wf1 = report1['weighted avg']['f1-score']

    # Stage 2: fine-tune with lower LR
    model.load_weights(str(output_dir / 'stage1_best.weights.h5'))

    train_ds2 = make_tf_dataset(x_train, y_train, max(16, args.batch_size // 2),
                                args.img_size, augment=True)

    print(f'\nStage 2: Fine-tuning (lr={args.lr_stage2}, {args.epochs_stage2} epochs)')
    hist2, val_acc2 = train_stage(
        model, train_ds2, val_ds, x_test, y_test,
        'stage2', args.lr_stage2, args.epochs_stage2, output_dir,
    )
    report2, cm2 = evaluate(model, x_test, y_test, CLASS_NAMES, output_dir, tag='stage2')
    wf2 = report2['weighted avg']['f1-score']

    # Summary
    results = {
        'model': f'mobilenetv5_{args.config}',
        'params': total,
        'val_acc_stage1': float(f'{val_acc1:.4f}'),
        'val_acc_stage2': float(f'{val_acc2:.4f}'),
        'test_wf1_stage1': float(f'{wf1:.4f}'),
        'test_wf1_stage2': float(f'{wf2:.4f}'),
        'report_stage1': {k: v for k, v in report1.items() if isinstance(v, dict)},
        'report_stage2': {k: v for k, v in report2.items() if isinstance(v, dict)},
    }
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n=== Results ===')
    print(f'  Stage 1: val_acc={val_acc1:.4f}, test_wF1={wf1:.4f}')
    print(f'  Stage 2: val_acc={val_acc2:.4f}, test_wF1={wf2:.4f}')
    print(f'  Results saved to {output_dir / "results.json"}')

    model.save(output_dir / 'model.keras')
    print('Done!')


if __name__ == '__main__':
    main()
