import logging
from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve


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


def _decision_inputs(probs, targets=None):
    probs_array = np.asarray(probs, dtype=float)
    if probs_array.ndim != 2 or probs_array.shape[1] < 2 or len(probs_array) == 0:
        raise ValueError("probs must be a non-empty two-dimensional array with at least two classes")
    if not np.all(np.isfinite(probs_array)) or np.any(probs_array < 0):
        raise ValueError("probs must contain finite non-negative values")
    if targets is None:
        return probs_array
    targets_array = np.asarray(targets)
    if targets_array.ndim != 1 or len(targets_array) != len(probs_array):
        raise ValueError("targets must be one-dimensional and match probs")
    return probs_array, targets_array.astype(int, copy=False)


def decide_argmax(probs) -> np.ndarray:
    return _decision_inputs(probs).argmax(axis=1)


def decide_prior_corrected(probs, train_priors, tau: float) -> np.ndarray:
    probs_array = _decision_inputs(probs)
    priors = np.asarray(train_priors, dtype=float)
    if priors.shape != (probs_array.shape[1],) or not np.all(np.isfinite(priors)) or np.any(priors <= 0):
        raise ValueError("train_priors must be positive, finite, and match the number of classes")
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("tau must be a finite non-negative value")
    scores = np.log(np.clip(probs_array, 1e-12, None)) - float(tau) * np.log(priors)
    return scores.argmax(axis=1)


def decide_malignant_gated(probs, threshold: float, malignant_indices: Sequence[int]) -> np.ndarray:
    probs_array = _decision_inputs(probs)
    malignant = np.asarray(list(malignant_indices), dtype=int)
    if len(malignant) == 0 or len(np.unique(malignant)) != len(malignant):
        raise ValueError("malignant_indices must contain unique class indices")
    if np.any((malignant < 0) | (malignant >= probs_array.shape[1])):
        raise ValueError("malignant_indices contain an out-of-range class index")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    benign = np.setdiff1d(np.arange(probs_array.shape[1]), malignant)
    gate = probs_array[:, malignant].sum(axis=1) >= threshold
    predictions = np.empty(len(probs_array), dtype=int)
    predictions[gate] = malignant[probs_array[gate][:, malignant].argmax(axis=1)]
    predictions[~gate] = benign[probs_array[~gate][:, benign].argmax(axis=1)]
    return predictions


def _balanced_accuracy(targets, predictions) -> float:
    labels = np.unique(targets)
    return float(np.mean([np.mean(predictions[targets == label] == label) for label in labels]))


def select_logit_adjust(probs_val, targets_val, train_priors, grid=None) -> tuple[float, list[dict]]:
    probs_array, targets_array = _decision_inputs(probs_val, targets_val)
    candidates = np.arange(0.0, 1.01, 0.1) if grid is None else np.asarray(grid, dtype=float)
    if candidates.ndim != 1 or len(candidates) == 0 or not np.all(np.isfinite(candidates)) or np.any(candidates < 0):
        raise ValueError("grid must contain finite non-negative values")
    table = []
    for tau in candidates:
        predictions = decide_prior_corrected(probs_array, train_priors, float(tau))
        table.append({
            "tau": round(float(tau), 4),
            "balanced_accuracy": _balanced_accuracy(targets_array, predictions),
            "macro_f1": float(f1_score(targets_array, predictions, labels=np.unique(targets_array), average="macro", zero_division=0)),
            "accuracy": float(np.mean(predictions == targets_array)),
        })
    best = max(table, key=lambda row: (row["balanced_accuracy"], row["macro_f1"], -row["tau"]))
    return float(best["tau"]), table


def confusion_summary(targets, predictions, class_names: Sequence[str], malignant_indices: Sequence[int] = (0, 1, 4), present_only: bool = True) -> dict:
    targets_array = np.asarray(targets, dtype=int)
    predictions_array = np.asarray(predictions, dtype=int)
    names = list(class_names)
    if targets_array.ndim != 1 or predictions_array.ndim != 1 or len(targets_array) != len(predictions_array) or len(targets_array) == 0:
        raise ValueError("targets and predictions must be non-empty matching one-dimensional arrays")
    if not names or np.any((targets_array < 0) | (targets_array >= len(names))) or np.any((predictions_array < 0) | (predictions_array >= len(names))):
        raise ValueError("class_names must cover all target and prediction indices")
    cm = confusion_matrix(targets_array, predictions_array, labels=range(len(names)))
    row_totals = cm.sum(axis=1)
    column_totals = cm.sum(axis=0)
    row_recall = np.divide(cm, row_totals[:, None], out=np.zeros_like(cm, dtype=float), where=row_totals[:, None] != 0)
    column_precision = np.divide(np.diag(cm), column_totals, out=np.zeros(len(names), dtype=float), where=column_totals != 0)
    present = np.flatnonzero(row_totals > 0)
    malignant = np.asarray(list(malignant_indices), dtype=int)
    true_malignant = np.isin(targets_array, malignant)
    pred_malignant = np.isin(predictions_array, malignant)
    tp = int(np.sum(true_malignant & pred_malignant))
    fn = int(np.sum(true_malignant & ~pred_malignant))
    fp = int(np.sum(~true_malignant & pred_malignant))
    tn = int(np.sum(~true_malignant & ~pred_malignant))
    div = lambda n, d: float(n / d) if d else 0.0
    selected_rows = present if present_only else np.arange(len(names))
    return {
        "counts": cm,
        "row_recall": row_recall,
        "column_precision": column_precision,
        "row_totals": row_totals,
        "present_indices": selected_rows,
        "balanced_accuracy": _balanced_accuracy(targets_array, predictions_array),
        "macro_f1": float(f1_score(targets_array, predictions_array, labels=present, average="macro", zero_division=0)),
        "accuracy": float(np.mean(targets_array == predictions_array)),
        "malignant": {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp), "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn)},
    }


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
