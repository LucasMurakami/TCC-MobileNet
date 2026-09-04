"""
Dataset Preparation and Data Loading Pipelines for HAM10000 and PAD-UFES-20.
Shared across all model training scripts (MobileNet V1, V2, V3, V4, and V5).
"""

from pathlib import Path
import hashlib
import os
import json
import shutil
import glob
import pandas as pd
from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

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
    columns = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in columns:
            return columns[candidate.lower()]
    return None


def grouped_stratified_split(
    df: pd.DataFrame,
    group_col: str,
    stratify_col: str,
    test_size: float,
    random_state: int,
):
    if group_col not in df.columns or stratify_col not in df.columns:
        missing = [column for column in (group_col, stratify_col) if column not in df.columns]
        raise KeyError(f"Missing required split columns: {missing}")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if df[group_col].isna().any():
        raise ValueError(f"{group_col} contains missing values")
    if df[stratify_col].isna().any():
        raise ValueError(f"{stratify_col} contains missing values")

    group_labels = df[[group_col, stratify_col]].drop_duplicates()
    inconsistent = group_labels.groupby(group_col)[stratify_col].nunique()
    if (inconsistent > 1).any():
        bad_groups = inconsistent[inconsistent > 1].index.astype(str).tolist()
        raise ValueError(f"Groups map to multiple classes: {bad_groups[:5]}")

    train_groups, test_groups = train_test_split(
        group_labels[group_col],
        test_size=test_size,
        random_state=random_state,
        stratify=group_labels[stratify_col],
    )
    train_group_set = set(train_groups)
    test_group_set = set(test_groups)
    train_df = df[df[group_col].isin(train_group_set)].copy().reset_index(drop=True)
    test_df = df[df[group_col].isin(test_group_set)].copy().reset_index(drop=True)
    if train_group_set & test_group_set:
        raise RuntimeError("Grouped split produced overlapping groups")
    return train_df, test_df


def validate_image_paths(df: pd.DataFrame):
    if 'path' not in df.columns:
        raise KeyError("DataFrame must contain a 'path' column")
    valid_indices = []
    bad_paths = []
    for index, value in df['path'].items():
        path = str(value)
        try:
            with Image.open(path) as image:
                image.verify()
            valid_indices.append(index)
        except (OSError, ValueError, TypeError):
            bad_paths.append(path)
    return df.loc[valid_indices].copy().reset_index(drop=True), bad_paths


def _image_ids_sha256(df: pd.DataFrame, image_id_col: str) -> str:
    if image_id_col not in df.columns:
        raise KeyError(f"DataFrame must contain '{image_id_col}'")
    image_ids = sorted(df[image_id_col].dropna().astype(str).tolist())
    return hashlib.sha256('\n'.join(image_ids).encode('utf-8')).hexdigest()


def build_split_manifest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int,
    val_size: float = 0.10,
    test_size: float = 0.20,
    group_col: str = 'lesion_id',
    stratify_col: str = 'dx',
    image_id_col: str = 'image_id',
):
    partitions = {'train': train_df, 'val': val_df, 'test': test_df}
    group_sets = {}
    partition_details = {}
    for name, frame in partitions.items():
        for column in (group_col, stratify_col, image_id_col):
            if column not in frame.columns:
                raise KeyError(f"{name} DataFrame must contain '{column}'")
        group_sets[name] = set(frame[group_col].dropna().astype(str))
        partition_details[name] = {
            'samples': int(len(frame)),
            'groups': int(frame[group_col].nunique()),
            'class_counts': {
                str(label): int(count)
                for label, count in frame[stratify_col].value_counts().sort_index().items()
            },
            'image_ids_sha256': _image_ids_sha256(frame, image_id_col),
        }

    overlap = {
        'train_val': len(group_sets['train'] & group_sets['val']),
        'train_test': len(group_sets['train'] & group_sets['test']),
        'val_test': len(group_sets['val'] & group_sets['test']),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Lesion leakage detected between partitions: {overlap}")
    all_image_ids = pd.concat([frame[image_id_col] for frame in partitions.values()], ignore_index=True)
    return {
        'seed': int(random_state),
        'random_state': int(random_state),
        'val_size': float(val_size),
        'test_size': float(test_size),
        'split_sizes': {name: details['samples'] for name, details in partition_details.items()},
        'class_counts': {name: details['class_counts'] for name, details in partition_details.items()},
        'group_counts': {name: details['groups'] for name, details in partition_details.items()},
        'group_overlap': overlap,
        'image_ids_sha256': {
            **{name: details['image_ids_sha256'] for name, details in partition_details.items()},
            'all': hashlib.sha256(
                '\n'.join(sorted(all_image_ids.dropna().astype(str).tolist())).encode('utf-8')
            ).hexdigest(),
        },
        'partitions': partition_details,
    }


def save_split_manifest(manifest: dict, output_path: Path) -> Path:
    output_path = Path(output_path)
    if output_path.suffix.lower() != '.json':
        output_path = output_path / 'split_manifest.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write('\n')
    return output_path


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


def load_pad_ufes20_validation(
    val_root: Path,
    label_mapping_path: Path = None,
) -> pd.DataFrame:
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
    lesion_id_col = _first_existing_column(
        df,
        ['lesion_id', 'lesion', 'lesion_identifier', 'lesion_number', 'lesion_no']
    )
    patient_id_col = _first_existing_column(
        df,
        ['patient_id', 'patient', 'patient_identifier', 'patient_number', 'patient_no']
    )
    path_col = _first_existing_column(df, ['path', 'filepath', 'file_path'])

    if path_col is not None:
        candidate_paths = df[path_col].map(
            lambda value: None if pd.isna(value) or not str(value).strip() else Path(str(value))
        )
        df['path'] = candidate_paths.map(
            lambda path: None if path is None else str(path if path.is_absolute() else val_root / path)
        )
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

    if image_id_col is not None:
        image_ids = df[image_id_col].astype('string')
    else:
        image_ids = df['path'].map(lambda value: Path(str(value)).stem).astype('string')
    missing_image_ids = image_ids.isna() | image_ids.str.strip().eq('')
    image_ids = image_ids.mask(missing_image_ids, df.index.map(lambda index: f"pad-image-{index}"))
    df['image_id'] = image_ids.astype(str)

    if lesion_id_col is not None:
        lesion_ids = df[lesion_id_col].astype('string')
        missing_lesion_ids = lesion_ids.isna() | lesion_ids.str.strip().eq('')
        lesion_ids = lesion_ids.mask(missing_lesion_ids, 'pad-lesion-' + df['image_id'])
    else:
        lesion_ids = 'pad-lesion-' + df['image_id']
    df['lesion_id'] = lesion_ids.astype(str)

    if patient_id_col is not None:
        patient_ids = df[patient_id_col].astype('string')
        missing_patient_ids = patient_ids.isna() | patient_ids.str.strip().eq('')
        patient_ids = patient_ids.mask(missing_patient_ids, 'pad-patient-' + df['lesion_id'])
    else:
        patient_ids = 'pad-patient-' + df['lesion_id']
    df['patient_id'] = patient_ids.astype(str)

    df['raw_label'] = df[diagnosis_col]
    normalized_labels = df['raw_label'].astype(str).str.strip().str.lower()
    df['dx'] = normalized_labels.map(PAD_UFES20_LABEL_MAP)
    path_available = df['path'].notna() & df['path'].astype(str).str.strip().ne('')
    kept_mask = df['dx'].notna() & path_available
    mapping_rows = []
    for raw_label, indices in normalized_labels.groupby(normalized_labels).groups.items():
        mapped_to = PAD_UFES20_LABEL_MAP.get(raw_label)
        kept = int(kept_mask.loc[indices].sum())
        total = int(len(indices))
        mapping_rows.append({
            'raw_label': raw_label,
            'total': total,
            'mapped_to': mapped_to,
            'kept': kept,
            'dropped': total - kept,
        })
    mapping_rows.sort(key=lambda row: row['raw_label'])
    mapping_path = Path(label_mapping_path) if label_mapping_path is not None else val_root / 'pad_label_mapping.json'
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with mapping_path.open('w', encoding='utf-8') as file:
        json.dump({'labels': mapping_rows}, file, indent=2, sort_keys=True)
        file.write('\n')

    unsupported_rows = int(df['dx'].isna().sum())
    df = df.loc[kept_mask].copy()
    df['path'] = df['path'].astype(str)
    print(pd.DataFrame(mapping_rows).to_string(index=False))
    print(
        f"[dataset] PAD-UFES-20 validation loaded from {metadata_path} "
        f"with {len(df)} samples ({unsupported_rows} unsupported rows dropped)"
    )
    return df.reset_index(drop=True)


def prepare_dataset(
    cache_root: Path,
    prepared_dir: Path,
    val_size: float = 0.10,
    test_size: float = 0.20,
    random_state: int = 42,
    oversample: bool = False,
):
    """Prepare HAM10000 train, validation, and test sets with lesion isolation."""
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

    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be positive and sum to less than 1")

    df = pd.read_csv(raw_dir / "HAM10000_metadata.csv")
    required_columns = {'image_id', 'lesion_id', 'dx'}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise KeyError(f"HAM10000 metadata is missing required columns: {missing_columns}")

    imageid_path_dict = {}
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        for f in glob.glob(os.path.join(raw_dir, '*', ext)):
            imageid_path_dict[os.path.splitext(os.path.basename(f))[0]] = f
    df['path'] = df['image_id'].map(imageid_path_dict)
    df = df.dropna(subset=['path', 'dx', 'lesion_id', 'image_id']).copy().reset_index(drop=True)

    train_val_df, test_df = grouped_stratified_split(
        df,
        group_col='lesion_id',
        stratify_col='dx',
        test_size=test_size,
        random_state=random_state,
    )
    relative_val_size = val_size / (1.0 - test_size)
    train_df, val_df = grouped_stratified_split(
        train_val_df,
        group_col='lesion_id',
        stratify_col='dx',
        test_size=relative_val_size,
        random_state=random_state,
    )

    if oversample:
        print("[dataset] Oversampling training set...")
        max_size = train_df['dx'].value_counts().max()
        train_df = pd.concat([
            resample(group, replace=True, n_samples=max_size, random_state=random_state)
            for _, group in train_df.groupby('dx')
        ]).reset_index(drop=True)

    group_sets = [set(frame['lesion_id']) for frame in (train_df, val_df, test_df)]
    if group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2]:
        raise RuntimeError("Lesion leakage detected between dataset partitions")
    print(
        f"[dataset] Train samples: {len(train_df)}, Val samples: {len(val_df)}, "
        f"Test samples: {len(test_df)}; lesion overlap: 0"
    )
    return train_df, val_df, test_df


def prepare_dataset_with_external_validation(cache_root: Path, prepared_dir: Path, pad_ufes_dir: Path, random_state: int = 42, oversample: bool = False):
    """Prepare HAM10000 training data and PAD-UFES-20 validation data."""
    cache_root = Path(cache_root)
    prepared_dir = Path(prepared_dir)
    pad_ufes_dir = Path(pad_ufes_dir)

    train_df, _, _ = prepare_dataset(cache_root, prepared_dir, random_state=random_state, oversample=oversample)
    val_dir = ensure_pad_ufes20_download(cache_root) if not pad_ufes_dir.exists() else pad_ufes_dir
    val_df = load_pad_ufes20_validation(val_dir)
    return train_df, val_df


