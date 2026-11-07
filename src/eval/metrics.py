"""Binary detection metrics, implemented on numpy so they can be unit-tested
without torch.

Convention used throughout: **fake is the positive class** (label 1, higher
score = more likely fake). Everything downstream -- thresholds, AUC sign,
AP -- depends on that and only that.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

__all__ = [
    "accuracy",
    "auc",
    "average_precision",
    "accuracy_at_fpr",
    "best_threshold_accuracy",
    "binary_metrics",
    "confusion_counts",
]


def _as_1d(x) -> np.ndarray:
    return np.asarray(x).reshape(-1).astype(np.float64)


def accuracy(labels, scores, threshold: float = 0.5) -> float:
    y = _as_1d(labels)
    s = _as_1d(scores)
    if y.size == 0:
        return float("nan")
    return float(((s >= threshold).astype(np.int64) == y.astype(np.int64)).mean())


def confusion_counts(labels, scores, threshold: float = 0.5) -> Dict[str, int]:
    y = _as_1d(labels).astype(np.int64)
    pred = (_as_1d(scores) >= threshold).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def auc(labels, scores) -> float:
    """Area under the ROC curve via the rank-sum (Mann-Whitney) identity.

    Ties are handled by average ranks, which makes this equal to
    ``sklearn.metrics.roc_auc_score`` to numerical precision while keeping the
    dependency footprint at numpy only.
    """
    y = _as_1d(labels).astype(np.int64)
    s = _as_1d(scores)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(s.size, dtype=np.float64)
    i = 0
    while i < s.size:
        j = i + 1
        while j < s.size and s_sorted[j] == s_sorted[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0  # average 1-based rank
        i = j

    rank_pos = ranks[y == 1].sum()
    return float((rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(labels, scores) -> float:
    """Average precision (area under the precision-recall curve, step-wise).

    This is the definition matching ``sklearn.metrics.average_precision_score``
    (precision at each distinct score threshold, weighted by the recall gain),
    not the trapezoidal interpolated variant.
    """
    y = _as_1d(labels).astype(np.int64)
    s = _as_1d(scores)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    # keep only the last index of each tied score block
    s_sorted = s[order]
    last = np.r_[np.where(np.diff(s_sorted) != 0)[0], s_sorted.size - 1]
    p = precision[last]
    r = recall[last]
    ap = np.sum(np.diff(np.r_[0.0, r]) * p)
    return float(ap)


def accuracy_at_fpr(labels, scores, fpr_target: float = 0.05) -> float:
    """TPR (i.e. fake recall) at a threshold where FPR <= fpr_target."""
    y = _as_1d(labels).astype(np.int64)
    s = _as_1d(scores)
    neg = np.sort(s[y == 0])[::-1]
    if neg.size == 0:
        return float("nan")
    k = int(np.floor(fpr_target * neg.size))
    if k >= neg.size:
        thr = -np.inf
    elif k <= 0:
        thr = np.nextafter(neg[0], np.inf)
    else:
        thr = neg[k - 1]  # allow at most k negatives above the threshold
        thr = np.nextafter(thr, np.inf)
    return float((s[y == 1] >= thr).mean())


def best_threshold_accuracy(labels, scores) -> float:
    """Maximum accuracy over all thresholds (an oracle upper bound)."""
    y = _as_1d(labels).astype(np.int64)
    s = _as_1d(scores)
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    n = y.size
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    correct = tp + (n - np.arange(1, n + 1) - fp)
    return float(correct.max() / n)


def binary_metrics(labels, scores, threshold: float = 0.5) -> Dict[str, float]:
    """One-stop summary. Keys are stable so they can be dumped straight to CSV."""
    out = {
        "n": float(_as_1d(labels).size),
        "acc": accuracy(labels, scores, threshold),
        "auc": auc(labels, scores),
        "ap": average_precision(labels, scores),
        "acc_best": best_threshold_accuracy(labels, scores),
        "tpr@fpr5": accuracy_at_fpr(labels, scores, 0.05),
    }
    out.update({k: float(v) for k, v in
                confusion_counts(labels, scores, threshold).items()})
    return out
