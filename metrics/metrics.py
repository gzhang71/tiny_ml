import numpy as np


def _to_binary(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5):
    return y_true.astype(int), (y_pred >= threshold).astype(int)


def precision(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """TP / (TP + FP). Returns 0.0 when there are no positive predictions."""
    t, p = _to_binary(y_true, y_pred, threshold)
    tp = int(np.sum((p == 1) & (t == 1)))
    fp = int(np.sum((p == 1) & (t == 0)))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """TP / (TP + FN). Returns 0.0 when there are no actual positives."""
    t, p = _to_binary(y_true, y_pred, threshold)
    tp = int(np.sum((p == 1) & (t == 1)))
    fn = int(np.sum((p == 0) & (t == 1)))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Harmonic mean of precision and recall."""
    p = precision(y_true, y_pred, threshold)
    r = recall(y_true, y_pred, threshold)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Fraction of correct predictions."""
    t, p = _to_binary(y_true, y_pred, threshold)
    return float(np.mean(t == p))
