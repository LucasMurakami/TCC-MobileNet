import logging
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics import (
    _evaluate_binary_triage,
    bootstrap_ci,
    bootstrap_metric_ci,
    compute_classification_metrics,
    evaluate_binary_triage,
    expected_calibration_error,
    meets_auc_target,
    restricted_class_accuracy,
)


def test_binary_triage_public_api_and_float_threshold():
    probs = np.array([0.05, 0.20, 0.60, 0.90])
    targets = np.array([0, 1, 0, 1])

    result = evaluate_binary_triage(probs, targets, 0.50)

    assert _evaluate_binary_triage is evaluate_binary_triage
    assert result["threshold"] == 0.5
    assert result["threshold_source"] == "float"
    assert result["threshold_fallback"] is False
    assert result["threshold_clamped"] is False
    assert result["sensitivity"] == 0.5
    assert result["specificity"] == 0.5
    assert result["detected"] == "1/2"
    assert len(result["operating_points"]) == 7


def test_binary_triage_threshold_provenance(caplog):
    probs = np.array([0.10, 0.20, 0.80, 0.90])
    targets = np.array([0, 0, 1, 1])

    assert evaluate_binary_triage(probs, targets, "youden")["threshold_source"] == "youden"
    assert evaluate_binary_triage(probs, targets, "sens90")["threshold_source"] == "sens90"

    with caplog.at_level(logging.WARNING):
        fallback = evaluate_binary_triage(probs, targets, "invalid", default_th=0.25)
        clamped = evaluate_binary_triage(probs, targets, 1.5)

    assert fallback["threshold_source"] == "fallback_default"
    assert fallback["threshold_fallback"] is True
    assert fallback["threshold"] == 0.25
    assert clamped["threshold_source"] == "clamped"
    assert clamped["threshold_clamped"] is True
    assert clamped["threshold"] == 0.9
    assert "fell back" in caplog.text
    assert "clamped" in caplog.text


def test_compute_classification_metrics():
    class_names = ["a", "b", "c"]
    targets = np.array([0, 1, 2, 0, 1, 2])
    probs = np.array([
        [0.90, 0.05, 0.05],
        [0.05, 0.90, 0.05],
        [0.05, 0.05, 0.90],
        [0.80, 0.10, 0.10],
        [0.10, 0.80, 0.10],
        [0.10, 0.10, 0.80],
    ])

    result = compute_classification_metrics(probs, targets, class_names)

    assert result["accuracy"] == 1.0
    assert result["weighted_avg_f1"] == 1.0
    assert result["macro_avg_f1"] == 1.0
    assert result["macro_auc_roc"] == 1.0
    assert result["per_class_recall"] == {"a": 1.0, "b": 1.0, "c": 1.0}
    assert result["per_class_auc"] == {"a": 1.0, "b": 1.0, "c": 1.0}
    np.testing.assert_array_equal(result["predictions"], targets)


def test_expected_calibration_error():
    targets = np.array([0, 1, 0, 1])
    perfect = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    uncertain = np.full((4, 2), 0.5)

    assert expected_calibration_error(perfect, targets) == pytest.approx(0.0)
    assert expected_calibration_error(uncertain, targets) == pytest.approx(0.0)


def test_restricted_class_accuracy_uses_restricted_argmax():
    probs = np.array([
        [0.40, 0.10, 0.50],
        [0.20, 0.30, 0.50],
        [0.10, 0.80, 0.10],
    ])
    targets = np.array([0, 1, 2])

    assert restricted_class_accuracy(probs, targets, [0, 1]) == 1.0
    assert restricted_class_accuracy(probs, targets, [0]) == 1.0
    assert restricted_class_accuracy(probs[:1], targets[:1], [1]) == 0.0


def binary_auc(probs, targets, class_index):
    binary_targets = targets == class_index
    if len(np.unique(binary_targets)) < 2:
        raise ValueError("AUC requires both binary classes")
    return roc_auc_score(binary_targets, probs[:, class_index])


def binary_sensitivity(probs, targets, class_index, threshold):
    binary_targets = targets == class_index
    if not np.any(binary_targets):
        raise ValueError("Sensitivity requires a positive sample")
    predictions = probs[:, class_index] >= threshold
    return np.sum(predictions & binary_targets) / np.sum(binary_targets)


def test_cluster_bootstrap_is_deterministic_for_auc_and_sensitivity():
    targets = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    positive_probs = np.array([0.10, 0.30, 0.70, 0.90, 0.20, 0.40, 0.60, 0.80])
    probs = np.column_stack([1.0 - positive_probs, positive_probs])
    auc_metric = partial(binary_auc, class_index=1)
    sensitivity_metric = partial(binary_sensitivity, class_index=1, threshold=0.75)

    first = bootstrap_ci(probs, targets, groups, auc_metric, n=100, seed=7)
    second = bootstrap_ci(probs, targets, groups, auc_metric, n=100, seed=7)
    sensitivity_ci = bootstrap_metric_ci(
        probs, targets, groups, sensitivity_metric, n=100, seed=7
    )

    assert np.asarray(first).shape == (2,)
    assert first == second
    assert first == pytest.approx((1.0, 1.0))
    assert 0.0 <= sensitivity_ci[0] <= sensitivity_ci[1] <= 1.0


def test_auc_target_uses_confidence_interval_lower_bound():
    assert meets_auc_target(0.85)
    assert not meets_auc_target(0.8499)
    assert not meets_auc_target(float('nan'))
