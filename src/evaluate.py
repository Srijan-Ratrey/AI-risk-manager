"""Metrics. Deliberately not accuracy.

PR-AUC over ROC-AUC: at a 3.5% base rate, ROC-AUC is dominated by the huge
negative class and stays flattering even when precision is unusable. Average
precision tracks what a reviewer actually experiences.

Accuracy appears exactly once in this project -- beside the never-fraud
baseline, to show why it is not used.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(y_true, scores))


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Reported only for comparison with published leaderboards."""
    return float(roc_auc_score(y_true, scores))


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.005) -> float:
    """How much fraud we catch while disturbing <= target_fpr of good customers.

    The most operationally meaningful single number in the whole report.
    """
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.interp(target_fpr, fpr, tpr))


def precision_recall_at(
    y_true: np.ndarray, blocked: np.ndarray
) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(bool)
    blocked = np.asarray(blocked).astype(bool)
    tp = int((y_true & blocked).sum())
    fp = int((~y_true & blocked).sum())
    fn = int((y_true & ~blocked).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 20
) -> float:
    """Weighted average gap between predicted probability and observed rate.

    Quantile bins, not equal-width: with a 3.5% base rate almost every
    prediction lands in the first equal-width bin and the metric goes blind.
    """
    y_true = np.asarray(y_true).astype(float)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    edges = np.unique(np.quantile(probabilities, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return 0.0

    bin_ids = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, len(edges) - 2)
    total = 0.0
    for b in range(len(edges) - 1):
        mask = bin_ids == b
        if not mask.any():
            continue
        total += mask.mean() * abs(probabilities[mask].mean() - y_true[mask].mean())
    return float(total)


def reliability_curve(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 20
) -> dict[str, np.ndarray]:
    """Points for the reliability diagram, on quantile bins."""
    y_true = np.asarray(y_true).astype(float)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    edges = np.unique(np.quantile(probabilities, np.linspace(0, 1, n_bins + 1)))
    bin_ids = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, len(edges) - 2)

    predicted, observed, weight = [], [], []
    for b in range(len(edges) - 1):
        mask = bin_ids == b
        if not mask.any():
            continue
        predicted.append(probabilities[mask].mean())
        observed.append(y_true[mask].mean())
        weight.append(mask.sum())
    return {
        "predicted": np.array(predicted),
        "observed": np.array(observed),
        "count": np.array(weight),
    }
