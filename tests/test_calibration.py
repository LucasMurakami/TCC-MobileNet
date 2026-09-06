import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from metrics import expected_calibration_error, fit_temperature, negative_log_likelihood, softmax_with_temperature


def _calibrated_logits(seed=0, n=4000, classes=7):
    """Targets are sampled from softmax(logits), so T=1 is the calibrated temperature by construction."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 2.0, size=(n, classes))
    probs = softmax_with_temperature(logits, 1.0)
    targets = np.array([rng.choice(classes, p=p) for p in probs])
    return logits, targets


def test_temperature_reduces_nll_and_ece_and_keeps_ranking():
    logits, targets = _calibrated_logits()
    overconfident = logits * 3.0
    temperature = fit_temperature(overconfident, targets)
    assert temperature == pytest.approx(3.0, rel=0.15)
    assert negative_log_likelihood(overconfident, targets, temperature) < negative_log_likelihood(overconfident, targets, 1.0)
    before = softmax_with_temperature(overconfident, 1.0)
    after = softmax_with_temperature(overconfident, temperature)
    assert expected_calibration_error(after, targets) < expected_calibration_error(before, targets)
    np.testing.assert_array_equal(after.argmax(1), before.argmax(1))
    for c in range(logits.shape[1]):
        assert roc_auc_score(targets == c, after[:, c]) == pytest.approx(roc_auc_score(targets == c, before[:, c]), abs=0.01)


def test_temperature_is_near_one_for_calibrated_logits():
    logits, targets = _calibrated_logits(seed=1)
    calibrated_t = fit_temperature(logits, targets)
    assert 0.9 < calibrated_t < 1.1


def test_temperature_bounds_and_validation():
    logits, targets = _calibrated_logits(n=50)
    assert 0.5 <= fit_temperature(logits, targets, t_min=0.5, t_max=5.0) <= 5.0
    with pytest.raises(ValueError):
        fit_temperature(logits[:, 0], targets)
    with pytest.raises(ValueError):
        softmax_with_temperature(logits, 0.0)
