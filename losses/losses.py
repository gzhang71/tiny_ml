import numpy as np
from tiny_ml.core.module import Loss


class MSELoss(Loss):
    """Mean squared error — standard regression loss."""

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self._diff = pred - target
        return float(np.mean(self._diff ** 2))

    def backward(self) -> np.ndarray:
        return 2.0 * self._diff / self._diff.size


class SoftmaxCrossEntropy(Loss):
    """Fused softmax + cross-entropy for numerical stability.

    Accepts integer class labels (N,) or one-hot targets (N, C).
    """

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        self._probs = exp_x / exp_x.sum(axis=-1, keepdims=True)

        n = logits.shape[0]
        if targets.ndim == 1:
            self._one_hot = np.zeros_like(self._probs)
            self._one_hot[np.arange(n), targets.astype(int)] = 1.0
        else:
            self._one_hot = targets

        return float(-np.sum(self._one_hot * np.log(self._probs + 1e-12)) / n)

    def backward(self) -> np.ndarray:
        n = self._probs.shape[0]
        return (self._probs - self._one_hot) / n


class BinaryCrossEntropy(Loss):
    """Element-wise binary cross-entropy — expects sigmoid-activated predictions."""

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self._pred = np.clip(pred, 1e-9, 1.0 - 1e-9)
        self._target = target
        return float(-np.mean(
            target * np.log(self._pred) + (1.0 - target) * np.log(1.0 - self._pred)
        ))

    def backward(self) -> np.ndarray:
        n = self._pred.size
        return (-(self._target / self._pred) + (1.0 - self._target) / (1.0 - self._pred)) / n
