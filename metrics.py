import logging
from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, roc_curve


logger = logging.getLogger(__name__)
_THRESHOLD_MIN = 0.01
_THRESHOLD_MAX = 0.90


def _as_binary_inputs(probs, targets):
    probs_array = np.asarray(probs, dtype=float)
    targets_array = np.asarray(targets)
    if probs_array.ndim != 1 or targets_array.ndim != 1:
        raise ValueError("probs and targets must be one-dimensional")
    if len(probs_array) != len(targets_array):
        raise ValueError("probs and targets must have the same length")
    if len(probs_array) == 0:
        raise ValueError("probs and targets must not be empty")
    if not np.all(np.isfinite(probs_array)):
        raise ValueError("probs must contain only finite values")
    if not np.all(np.isin(targets_array, [0, 1])):
        raise ValueError("targets must contain only 0 and 1")
    return probs_array, targets_array.astype(int, copy=False)


def evaluate_binary_triage(probs, targets, threshold_spec=None, default_th: float = 0.15) -> dict:
    probs_array, targets_array = _as_binary_inputs(probs, targets)
    operating_points = {}
    total_pos = int(targets_array.sum())
    for ref_th in [0.50, 0.30, 0.20, 0.15, 0.10, 0.05, 0.02]:
        bin_preds = (probs_array >= ref_th).astype(int)
        tp = int(np.sum((bin_preds == 1) & (targets_array == 1)))
        fn = int(np.sum((bin_preds == 0) & (targets_array == 1)))
        fp = int(np.sum((bin_preds == 1) & (targets_array == 0)))
        tn = int(np.sum((bin_preds == 0) & (targets_array == 0)))
        sensitivity = float(tp / (tp + fn)) if tp + fn else 0.0
        specificity = float(tn / (tn + fp)) if tn + fp else 0.0
        operating_points[f"tau_{ref_th:.2f}"] = {
            "threshold": ref_th,
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "detected": f"{tp}/{total_pos}",
        }

    effective_th = float(default_th)
    threshold_source = "fallback_default"
    threshold_fallback = threshold_spec is None
    threshold_clamped = False

    if threshold_spec is not None:
        if isinstance(threshold_spec, str):
            normalized_spec = threshold_spec.lower().strip()
            if normalized_spec in ("auto", "youden"):
                if len(np.unique(targets_array)) > 1:
                    fpr, tpr, thresholds = roc_curve(targets_array, probs_array)
                    effective_th = float(thresholds[np.argmax(tpr - fpr)])
                    threshold_source = "youden"
                    threshold_fallback = False
                else:
                    threshold_fallback = True
            elif normalized_spec in ("sens90", "sens_90", "recall90"):
                if len(np.unique(targets_array)) > 1:
                    _, tpr, thresholds = roc_curve(targets_array, probs_array)
                    valid = np.flatnonzero(tpr >= 0.90)
                    if len(valid):
                        effective_th = float(thresholds[valid[0]])
                        threshold_source = "sens90"
                        threshold_fallback = False
                    else:
                        threshold_fallback = True
                else:
                    threshold_fallback = True
            elif normalized_spec in ("sens95", "sens_95", "recall95"):
                if len(np.unique(targets_array)) > 1:
                    _, tpr, thresholds = roc_curve(targets_array, probs_array)
                    valid = np.flatnonzero(tpr >= 0.95)
                    if len(valid):
                        effective_th = float(thresholds[valid[0]])
                        threshold_source = "sens95"
                        threshold_fallback = False
                    else:
                        threshold_fallback = True
                else:
                    threshold_fallback = True
            else:
                try:
                    effective_th = float(normalized_spec)
                    threshold_source = "float"
                    threshold_fallback = False
                except ValueError:
                    threshold_fallback = True
        else:
            try:
                effective_th = float(threshold_spec)
                threshold_source = "float"
                threshold_fallback = False
            except (TypeError, ValueError):
                threshold_fallback = True

    if threshold_fallback:
        effective_th = float(default_th)
        threshold_source = "fallback_default"
        if threshold_spec is not None:
            logger.warning("Binary triage threshold fell back to default %.4f", effective_th)

    clamped_th = float(np.clip(effective_th, _THRESHOLD_MIN, _THRESHOLD_MAX))
    if not np.isfinite(effective_th) or clamped_th != effective_th:
        if not np.isfinite(effective_th):
            clamped_th = float(default_th)
        threshold_clamped = True
        threshold_source = "clamped"
        logger.warning("Binary triage threshold was clamped from %s to %.4f", effective_th, clamped_th)
    effective_th = clamped_th

    bin_preds = (probs_array >= effective_th).astype(int)
    tp = int(np.sum((bin_preds == 1) & (targets_array == 1)))
    fn = int(np.sum((bin_preds == 0) & (targets_array == 1)))
    fp = int(np.sum((bin_preds == 1) & (targets_array == 0)))
    tn = int(np.sum((bin_preds == 0) & (targets_array == 0)))
    sensitivity = float(tp / (tp + fn)) if tp + fn else 0.0
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    f1 = float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0

    return {
        "threshold": round(effective_th, 4),
        "threshold_source": threshold_source,
        "threshold_fallback": threshold_fallback,
        "threshold_clamped": threshold_clamped,
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "detected": f"{tp}/{total_pos}",
        "operating_points": operating_points,
    }


_evaluate_binary_triage = evaluate_binary_triage


def compute_classification_metrics(probs, targets, class_names: Sequence[str]) -> dict:
    probs_array = np.asarray(probs, dtype=float)
    targets_array = np.asarray(targets)
    names = list(class_names)
    if probs_array.ndim != 2:
        raise ValueError("probs must be a two-dimensional array")
    if targets_array.ndim != 1 or len(targets_array) != len(probs_array):
        raise ValueError("targets must be one-dimensional and match probs")
    if probs_array.shape[1] != len(names) or not names:
        raise ValueError("class_names must match the number of probability columns")
    if len(targets_array) == 0:
        raise ValueError("probs and targets must not be empty")
    if not np.all(np.isfinite(probs_array)):
        raise ValueError("probs must contain only finite values")
    if not np.issubdtype(targets_array.dtype, np.integer):
        if not np.all(targets_array == targets_array.astype(int)):
            raise ValueError("targets must contain integer class indices")
        targets_array = targets_array.astype(int)
    if np.any((targets_array < 0) | (targets_array >= len(names))):
        raise ValueError("targets contain an out-of-range class index")

    predictions = probs_array.argmax(axis=1)
    report = classification_report(
        targets_array,
        predictions,
        labels=range(len(names)),
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    per_class_auc = {}
    valid_aucs = []
    for class_index, class_name in enumerate(names):
        binary_targets = (targets_array == class_index).astype(int)
        if len(np.unique(binary_targets)) > 1:
            auc_score = float(roc_auc_score(binary_targets, probs_array[:, class_index]))
            per_class_auc[class_name] = auc_score
            valid_aucs.append(auc_score)
        else:
            per_class_auc[class_name] = 0.0

    per_class_recall = {name: float(report[name]["recall"]) for name in names}
    metrics = {
        "accuracy": float(report["accuracy"]),
        "weighted_avg_f1": float(report["weighted avg"]["f1-score"]),
        "macro_avg_f1": float(report["macro avg"]["f1-score"]),
        "per_class_recall": per_class_recall,
        "per_class_auc": per_class_auc,
        "macro_auc_roc": float(np.mean(valid_aucs)) if valid_aucs else 0.0,
        "predictions": predictions,
        "report": report,
    }
    metrics.update({f"{name}_recall": recall for name, recall in per_class_recall.items()})
    metrics.update({f"{name}_auc_roc": score for name, score in per_class_auc.items()})
    return metrics


def expected_calibration_error(probs, targets, n_bins: int = 15) -> float:
    probs_array = np.asarray(probs, dtype=float)
    targets_array = np.asarray(targets)
    if probs_array.ndim != 2 or targets_array.ndim != 1 or len(probs_array) != len(targets_array):
        raise ValueError("targets must be one-dimensional and match two-dimensional probs")
    if len(targets_array) == 0 or n_bins < 1:
        raise ValueError("inputs must not be empty and n_bins must be positive")
    confidence = probs_array.max(axis=1)
    predictions = probs_array.argmax(axis=1)
    correct = predictions == targets_array
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & (confidence < upper if index < n_bins - 1 else confidence <= upper)
        if np.any(mask):
            error += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return error


def restricted_class_accuracy(probs, targets, class_indices: Sequence[int]) -> float:
    probs_array = np.asarray(probs, dtype=float)
    targets_array = np.asarray(targets)
    indices = np.asarray(list(class_indices), dtype=int)
    if probs_array.ndim != 2:
        raise ValueError("probs must be a two-dimensional array")
    if targets_array.ndim != 1 or len(targets_array) != len(probs_array):
        raise ValueError("targets must be one-dimensional and match probs")
    if indices.ndim != 1 or len(indices) == 0 or len(np.unique(indices)) != len(indices):
        raise ValueError("class_indices must contain unique class indices")
    if np.any((indices < 0) | (indices >= probs_array.shape[1])):
        raise ValueError("class_indices contain an out-of-range class index")
    mask = np.isin(targets_array, indices)
    if not np.any(mask):
        return 0.0
    restricted_predictions = indices[np.argmax(probs_array[mask][:, indices], axis=1)]
    return float(np.mean(restricted_predictions == targets_array[mask]))


def bootstrap_ci(
    probs,
    targets,
    groups,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    probs_array = np.asarray(probs)
    targets_array = np.asarray(targets)
    groups_array = np.asarray(groups)
    if probs_array.ndim not in (1, 2):
        raise ValueError("probs must be one- or two-dimensional")
    if targets_array.ndim != 1 or groups_array.ndim != 1:
        raise ValueError("targets and groups must be one-dimensional")
    if len(probs_array) != len(targets_array) or len(targets_array) != len(groups_array):
        raise ValueError("probs, targets, and groups must have the same length")
    if len(targets_array) == 0:
        raise ValueError("probs, targets, and groups must not be empty")
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not callable(metric_fn):
        raise TypeError("metric_fn must be callable")

    unique_groups = np.unique(groups_array)
    group_rows = [np.flatnonzero(groups_array == group) for group in unique_groups]
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        sampled_groups = rng.integers(0, len(unique_groups), size=len(unique_groups))
        sampled_rows = np.concatenate([group_rows[index] for index in sampled_groups])
        try:
            score = float(metric_fn(probs_array[sampled_rows], targets_array[sampled_rows]))
        except ValueError:
            continue
        if np.isfinite(score):
            scores.append(score)
    if not scores:
        return float("nan"), float("nan")
    low, high = np.percentile(np.asarray(scores), [2.5, 97.5])
    return float(low), float(high)


def meets_auc_target(ci_low: float, target: float = 0.85) -> bool:
    return bool(np.isfinite(ci_low) and ci_low >= target)


def bootstrap_metric_ci(
    probs,
    targets,
    groups,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    return bootstrap_ci(probs, targets, groups, metric_fn, n=n, seed=seed)
