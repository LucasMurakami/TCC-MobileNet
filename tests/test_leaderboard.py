import json

import pandas as pd

from main import update_session_leaderboard


def _write_result(path, sha, seed, ham_auc, pad_auc):
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        'model': 'v1', 'seed': seed, 'checkpoint_sha256': sha, 'params': 3_200_000, 'selected_epoch': 5,
        'ham_accuracy': 0.8, 'ham_test_mel_auc_roc': ham_auc, 'ham_mel_auc_roc': ham_auc, 'pad_mel_auc_roc': pad_auc,
        'pad_macro_auc_roc': 0.7, 'ham_balanced_accuracy': 0.66, 'ham_balanced_accuracy_tau': 0.73,
        'pad_balanced_accuracy': 0.35, 'pad_balanced_accuracy_tau': 0.43, 'selected_logit_adjust': 0.5,
        'logit_adjust_source': 'ham_val_balanced_accuracy', 'pad_malignant_gated_sensitivity': 0.55,
        'ham_expected_calibration_error': 0.09, 'pad_expected_calibration_error': 0.3, 'temperature': 1.4,
        'ham_val_accuracy': 0.85, 'selected_epoch_loop_accuracy': 0.85, 'ham_val_accuracy_delta': 0.0,
    }
    with open(path / 'results.json', 'w') as f:
        json.dump(payload, f)


def test_seeded_layout_aggregates_and_dedups(tmp_path):
    base = tmp_path / 'scenarios' / 'main' / 'v1'
    _write_result(base / 'seed42', 'aaa', 42, 0.90, 0.80)
    _write_result(base / 'seed43', 'bbb', 43, 0.88, 0.78)
    _write_result(tmp_path / 'scenarios' / 'legacy' / 'v1', 'aaa', 42, 0.90, 0.80)

    df = update_session_leaderboard(tmp_path)

    assert len(df) == 3
    assert sorted(df['seed'].tolist()) == [42, 42, 43]
    assert df['duplicate_of'].notna().sum() == 1
    assert (tmp_path / 'SUMMARY.md').exists()
    summary = (tmp_path / 'SUMMARY.md').read_text(encoding='utf-8')
    assert '3 run(s), 2 unique checkpoint(s)' in summary
    assert '| main | v1 | 2 |' in summary
    assert '0.8900 ± 0.0100' in summary
    csv = pd.read_csv(tmp_path / 'master_leaderboard.csv')
    assert set(['seed', 'run', 'duplicate_of', 'temperature', 'params_m']).issubset(csv.columns)
