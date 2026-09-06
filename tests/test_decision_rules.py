import numpy as np
import pytest

from visualize import plot_confusion_matrices, plot_decision_confusion_matrices
from metrics import (
    confusion_summary,
    decide_argmax,
    decide_malignant_gated,
    decide_prior_corrected,
    select_logit_adjust,
)


def test_prior_correction_invariants():
    probs = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2]])
    np.testing.assert_array_equal(decide_prior_corrected(probs, [0.7, 0.2, 0.1], 0), decide_argmax(probs))
    np.testing.assert_array_equal(decide_prior_corrected(probs, [1 / 3] * 3, 1), decide_argmax(probs))


def test_malignant_gate_respects_selected_group():
    probs = np.array([[0.2, 0.25, 0.3, 0.25], [0.1, 0.1, 0.7, 0.1]])
    predictions = decide_malignant_gated(probs, 0.4, [0, 1])
    assert predictions[0] in (0, 1)
    assert predictions[1] in (2, 3)


def test_select_logit_adjust_prefers_lower_tau_on_tie():
    probs = np.array([[0.8, 0.2], [0.2, 0.8]])
    tau, table = select_logit_adjust(probs, np.array([0, 1]), [0.5, 0.5], grid=[0, 0.5, 1])
    assert tau == 0
    assert len(table) == 3


def test_confusion_summary_handles_absent_classes():
    summary = confusion_summary([0, 0, 1, 1], [0, 1, 1, 1], ["a", "b", "absent"], malignant_indices=[1])
    np.testing.assert_array_equal(summary["present_indices"], [0, 1])
    assert np.isfinite(summary["row_recall"]).all()
    assert np.isfinite(summary["column_precision"]).all()
    assert summary["malignant"]["sensitivity"] == pytest.approx(1.0)


def test_confusion_plots_support_absent_classes(tmp_path):
    class_names = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    targets = np.array([0, 0, 1, 1, 2, 4, 5, 5])
    probs = np.full((len(targets), 7), 0.02)
    probs[np.arange(len(targets)), targets] = 0.88
    probs /= probs.sum(axis=1, keepdims=True)
    matrix_path = tmp_path / "matrix.png"
    decision_path = tmp_path / "decisions.png"
    plot_confusion_matrices(targets, probs.argmax(1), class_names, matrix_path, "test")
    plot_decision_confusion_matrices(probs, targets, class_names, np.full(7, 1 / 7), 0.5, 0.4, decision_path, "test")
    assert matrix_path.stat().st_size > 1000
    assert decision_path.stat().st_size > 1000
