import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from dataset import (
    PAD_UFES20_LABEL_MAP,
    build_split_manifest,
    grouped_stratified_split,
    load_pad_ufes20_validation,
    prepare_dataset,
    save_split_manifest,
    validate_image_paths,
)


def synthetic_lesion_frame(groups_per_class=10, images_per_group=2):
    rows = []
    for label in ('akiec', 'bcc', 'mel'):
        for group_index in range(groups_per_class):
            lesion_id = f'{label}-lesion-{group_index}'
            for image_index in range(images_per_group):
                image_id = f'{lesion_id}-image-{image_index}'
                rows.append({
                    'lesion_id': lesion_id,
                    'image_id': image_id,
                    'dx': label,
                    'path': f'{image_id}.jpg',
                })
    return pd.DataFrame(rows)


def assert_disjoint_groups(*frames):
    group_sets = [set(frame['lesion_id']) for frame in frames]
    for index, left in enumerate(group_sets):
        for right in group_sets[index + 1:]:
            assert left.isdisjoint(right)


def test_grouped_stratified_split_preserves_groups_and_classes():
    frame = synthetic_lesion_frame()
    train_df, test_df = grouped_stratified_split(
        frame, 'lesion_id', 'dx', test_size=0.2, random_state=7
    )

    assert_disjoint_groups(train_df, test_df)
    assert set(train_df['dx']) == set(frame['dx'])
    assert set(test_df['dx']) == set(frame['dx'])
    assert len(train_df) + len(test_df) == len(frame)


def test_prepare_dataset_uses_three_way_grouped_split_and_ignores_ready(tmp_path):
    raw_dir = tmp_path / 'cache' / 'ham10000_raw'
    image_dir = raw_dir / 'images'
    image_dir.mkdir(parents=True)
    frame = synthetic_lesion_frame()
    frame.drop(columns='path').to_csv(raw_dir / 'HAM10000_metadata.csv', index=False)
    for image_id in frame['image_id']:
        (image_dir / f'{image_id}.jpg').touch()

    prepared_dir = tmp_path / 'legacy_prepared'
    prepared_dir.mkdir()
    (prepared_dir / '.ready').touch()

    train_df, val_df, test_df = prepare_dataset(
        tmp_path / 'cache', prepared_dir, random_state=11, oversample=False
    )

    assert [len(train_df), len(val_df), len(test_df)] == [42, 6, 12]
    assert_disjoint_groups(train_df, val_df, test_df)
    for split in (train_df, val_df, test_df):
        assert {'lesion_id', 'image_id', 'dx', 'path'} <= set(split.columns)


def test_validate_image_paths_is_explicit_and_returns_bad_paths(tmp_path):
    valid_path = tmp_path / 'valid.png'
    bad_path = tmp_path / 'bad.png'
    Image.new('RGB', (2, 2)).save(valid_path)
    bad_path.write_bytes(b'not-an-image')
    frame = pd.DataFrame({'path': [str(valid_path), str(bad_path)], 'image_id': ['ok', 'bad']})

    clean_df, bad_paths = validate_image_paths(frame)

    assert clean_df['image_id'].tolist() == ['ok']
    assert bad_paths == [str(bad_path)]


def test_pad_loader_preserves_raw_labels_and_stable_group_ids(tmp_path):
    raw_labels = ['ack', 'bcc', 'mel', 'nev', 'sek', 'scc']
    metadata = pd.DataFrame({
        'img_id': [f'pad-{index}' for index in range(len(raw_labels))],
        'diagnostic': raw_labels,
        'filepath': [f'images/pad-{index}.png' for index in range(len(raw_labels))],
        'lesion': ['shared', 'shared', None, 'lesion-3', 'lesion-4', 'lesion-5'],
        'patient': ['patient-0', 'patient-0', None, 'patient-3', 'patient-4', 'patient-5'],
    })
    metadata.to_csv(tmp_path / 'metadata.csv', index=False)

    first = load_pad_ufes20_validation(tmp_path)
    second = load_pad_ufes20_validation(tmp_path)

    assert first['raw_label'].tolist() == raw_labels
    assert first['dx'].tolist() == [PAD_UFES20_LABEL_MAP[label] for label in raw_labels]
    assert first['lesion_id'].tolist() == second['lesion_id'].tolist()
    assert first['patient_id'].tolist() == second['patient_id'].tolist()
    assert first.loc[0, 'lesion_id'] == 'shared'
    assert first.loc[0, 'patient_id'] == 'patient-0'
    assert first.loc[2, 'lesion_id'] == 'pad-lesion-pad-2'
    assert first.loc[2, 'patient_id'] == 'pad-patient-pad-lesion-pad-2'
    mapping = json.loads((tmp_path / 'pad_label_mapping.json').read_text(encoding='utf-8'))
    assert {row['raw_label'] for row in mapping['labels']} == set(raw_labels)
    assert all(row['dropped'] == 0 for row in mapping['labels'])


def test_split_manifest_records_counts_overlap_and_stable_hash(tmp_path):
    frame = synthetic_lesion_frame()
    train_val_df, test_df = grouped_stratified_split(
        frame, 'lesion_id', 'dx', test_size=0.2, random_state=5
    )
    train_df, val_df = grouped_stratified_split(
        train_val_df, 'lesion_id', 'dx', test_size=0.125, random_state=5
    )

    manifest = build_split_manifest(train_df, val_df, test_df, random_state=5)
    output_path = save_split_manifest(manifest, tmp_path)

    expected_hash = hashlib.sha256(
        '\n'.join(sorted(frame['image_id'])).encode('utf-8')
    ).hexdigest()
    assert manifest['random_state'] == 5
    assert manifest['split_sizes'] == {'train': 42, 'val': 6, 'test': 12}
    assert manifest['group_overlap'] == {'train_val': 0, 'train_test': 0, 'val_test': 0}
    assert manifest['image_ids_sha256']['all'] == expected_hash
    assert manifest['class_counts']['val'] == {'akiec': 2, 'bcc': 2, 'mel': 2}
    assert output_path == tmp_path / 'split_manifest.json'
    assert json.loads(output_path.read_text(encoding='utf-8')) == manifest


def test_split_manifest_rejects_lesion_overlap():
    frame = synthetic_lesion_frame()
    train_df = frame.iloc[:30].copy()
    val_df = frame.iloc[30:45].copy()
    test_df = frame.iloc[45:].copy()
    val_df.loc[val_df.index[0], 'lesion_id'] = train_df.iloc[0]['lesion_id']
    with pytest.raises(RuntimeError, match='Lesion leakage'):
        build_split_manifest(train_df, val_df, test_df, random_state=42)
