"""
Dataset Preparation and Data Loading Pipelines for HAM10000 and PAD-UFES-20.
Shared across all model training scripts (MobileNet V1-V4, MobileNetV5, Inception/Xception, etc.)
"""

from pathlib import Path
import os
import json
import shutil
import glob
import numpy as np
import pandas as pd
import tensorflow as tf

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, class_weight as sk_class_weight

# ─── Constants ───────────────────────────────────────────────────────────────
NUM_CLASSES = 7
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
HAM10000_CLASSES = {name: idx for idx, name in enumerate(CLASS_NAMES)}

PAD_UFES20_LABEL_MAP = {
    'ack': 'akiec',
    'akiec': 'akiec',
    'actinic keratosis': 'akiec',
    'scc': 'akiec',
    'squamous cell carcinoma': 'akiec',
    'bod': 'akiec',
    'bcc': 'bcc',
    'basal cell carcinoma': 'bcc',
    'mel': 'mel',
    'melanoma': 'mel',
    'nev': 'nv',
    'nevus': 'nv',
    'sek': 'bkl',
    'seborrheic keratosis': 'bkl',
}


def _first_existing_column(df: pd.DataFrame, candidates: list):
    for column in candidates:
        if column in df.columns:
            return column
    return None


def ensure_pad_ufes20_download(cache_root: Path) -> Path:
    """Check for PAD-UFES-20 locally or display instructions/links to download."""
    val_root = Path(cache_root) / 'pad_ufes_20_raw'
    csv_candidates = [
        val_root / 'metadata.csv',
        val_root / 'PADUFES20_metadata.csv',
        val_root / 'PAD-UFES-20_metadata.csv',
        val_root / 'PADUFES20.csv',
    ]
    if any(path.exists() for path in csv_candidates):
        return val_root

    val_root.mkdir(parents=True, exist_ok=True)

    bar = "=" * 78
    warning_msg = (
        f"\n{bar}\n"
        f" [!] WARNING: PAD-UFES-20 Dataset Not Found\n"
        f"{bar}\n"
        f"To use PAD-UFES-20 for external validation, please download the dataset from Mendeley Data:\n"
        f"  --> https://data.mendeley.com/datasets/zr7vgbcyr2/1\n\n"
        f"Extract the files into:\n"
        f"  --> {val_root.resolve()}/\n"
        f"The folder must contain 'metadata.csv' (or 'PADUFES20_metadata.csv') and the image files.\n"
        f"{bar}\n"
    )
    print(warning_msg)
    raise FileNotFoundError(
        f"PAD-UFES-20 dataset not found at {val_root}. "
        "Please download it from Mendeley Data (https://data.mendeley.com/datasets/zr7vgbcyr2/1) "
        f"and extract it to {val_root}."
    )


def load_pad_ufes20_validation(val_root: Path) -> pd.DataFrame:
    """Load PAD-UFES-20 as an external validation set.

    Diagnosis labels are mapped into the HAM10000 label space and unsupported
    classes are dropped.
    """
    val_root = Path(val_root)
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
        candidate_paths = df[path_col].astype(str).map(Path)
        if candidate_paths.map(lambda path: path.is_file()).all():
            df['path'] = candidate_paths.astype(str)
        else:
            df['path'] = candidate_paths.map(lambda path: path if path.is_absolute() else val_root / path)
            df['path'] = df['path'].map(str)
    elif image_id_col is not None:
        imageid_path_dict = {}
        for root, _, files in os.walk(str(val_root)):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in ('.png', '.jpg', '.jpeg'):
                    stem = os.path.splitext(f)[0]
                    full_p = os.path.join(root, f)
                    imageid_path_dict[stem] = full_p
                    imageid_path_dict[f] = full_p
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


def prepare_dataset(cache_root: Path, prepared_dir: Path, test_size: float = 0.2, random_state: int = 42):
    """Prepare HAM10000 training data and internal validation set."""
    cache_root = Path(cache_root)
    prepared_dir = Path(prepared_dir)
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
        train_df, val_df = train_test_split(df_local, test_size=test_size, random_state=random_state, stratify=df_local['dx'])
        return train_df, val_df

    # Split first
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['dx'])

    # Oversample training
    print("[dataset] Oversampling training set...")
    max_size = train_df['dx'].value_counts().max()
    lst = []
    for _, group in train_df.groupby('dx'):
        lst.append(resample(group, replace=True, n_samples=max_size, random_state=random_state))
    train_df = pd.concat(lst)

    print(f"[dataset] Train: {len(train_df)}, Val: {len(val_df)}")
    return train_df, val_df


def prepare_dataset_with_external_validation(cache_root: Path, prepared_dir: Path, pad_ufes_dir: Path, random_state: int = 42):
    """Prepare HAM10000 training data and PAD-UFES-20 validation data."""
    cache_root = Path(cache_root)
    prepared_dir = Path(prepared_dir)
    pad_ufes_dir = Path(pad_ufes_dir)

    train_df, _ = prepare_dataset(cache_root, prepared_dir, random_state=random_state)
    val_dir = ensure_pad_ufes20_download(cache_root) if not pad_ufes_dir.exists() else pad_ufes_dir
    val_df = load_pad_ufes20_validation(val_dir)
    return train_df, val_df


def get_data_generators(train_df: pd.DataFrame, val_df: pd.DataFrame, img_size: int = 224, batch_size: int = 32):
    """ImageDataGenerator pipeline for keras flow_from_dataframe."""
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
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
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_dataframe(
        train_df, x_col='path', y_col='dx', target_size=(img_size, img_size),
        batch_size=batch_size, class_mode='categorical', classes=CLASS_NAMES,
    )
    val_gen = val_datagen.flow_from_dataframe(
        val_df, x_col='path', y_col='dx', target_size=(img_size, img_size),
        batch_size=batch_size, class_mode='categorical', classes=CLASS_NAMES, shuffle=False,
    )
    return train_gen, val_gen


def make_tf_dataset(paths, labels, batch_size: int = 32, img_size: int = 224, augment: bool = False, shuffle: bool = True):
    """tf.data.Dataset pipeline for fast batched training."""
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


def compute_class_weights(y_train) -> dict:
    """Compute balanced class weights dictionary."""
    if isinstance(y_train, pd.Series):
        classes = np.sort(y_train.unique())
        weights = sk_class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
        return {i: float(w) for i, w in enumerate(weights)}
    else:
        classes = np.unique(y_train)
        weights = sk_class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
        return dict(zip(classes, weights))


# ─── Shared Training Utilities ───────────────────────────────────────────────

from time import perf_counter


class FocalLoss(tf.keras.losses.Loss):
    """Focal loss supporting both one-hot and integer targets."""
    def __init__(self, gamma=2.0, name='focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        if len(y_true.shape) == len(y_pred.shape):
            # Categorical / one-hot
            ce = -y_true * tf.math.log(y_pred)
            modulating = tf.pow(1.0 - y_pred, self.gamma)
            return tf.reduce_mean(modulating * ce)
        else:
            # Sparse / integer targets
            y_true_int = tf.cast(y_true, tf.int32)
            pt = tf.gather(y_pred, y_true_int, batch_dims=1)
            ce = -tf.math.log(pt)
            modulating = tf.pow(1.0 - pt, self.gamma)
            return tf.reduce_mean(modulating * ce)

    def get_config(self):
        config = super().get_config()
        config.update({'gamma': self.gamma})
        return config


class EpochTimingCallback(tf.keras.callbacks.Callback):
    """Callback to print elapsed time and metrics cleanly after each epoch."""
    def __init__(self, stage_name='train'):
        super().__init__()
        self.stage_name = stage_name
        self.epoch_start = None
        self.stage_start = None

    def on_train_begin(self, logs=None):
        self.stage_start = perf_counter()
        print(f"  [{self.stage_name}] training started")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = perf_counter() - self.epoch_start if self.epoch_start is not None else 0.0
        metrics = ' | '.join(
            f"{k}={v:.4f}" for k, v in (logs or {}).items() if isinstance(v, (int, float, np.floating, np.integer))
        )
        epoch_total = self.params.get('epochs', epoch + 1)
        suffix = f" | {metrics}" if metrics else ""
        print(f"  [{self.stage_name}] epoch {epoch + 1}/{epoch_total} took {elapsed:.1f}s{suffix}")

    def on_train_end(self, logs=None):
        if self.stage_start is not None:
            total_elapsed = perf_counter() - self.stage_start
            print(f"  [{self.stage_name}] training finished in {total_elapsed:.1f}s")

