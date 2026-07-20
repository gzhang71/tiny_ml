import numpy as onp  # shape bookkeeping only

from core.backend import xp as np
from core.module import Loss


class MSELoss(Loss):
    """Mean squared error — standard regression loss."""

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self._diff = pred - target
        return float(np.mean(self._diff ** 2))

    def backward(self) -> np.ndarray:
        return 2.0 * self._diff / self._diff.size


class SoftmaxCrossEntropy(Loss):
    """Fused softmax + cross-entropy for numerical stability.

    Given logits of shape (..., C), targets are accepted in either form:

    - **integer labels**, shape (...) — i.e. the logits shape without its class
      axis. `(N,)` for a plain batch, `(B, T)` for a language model.
    - **one-hot / soft targets**, shape (..., C) — exactly the logits shape.

    Which form was passed is decided by *shape*, and anything else raises.
    Sniffing `targets.ndim` instead (the obvious shortcut) silently misreads
    (B, T) integer labels as a one-hot array and returns a plausible-looking
    but meaningless number.

    The loss is the mean over every predicted position — all the leading dims,
    not just the batch axis — so (B, T, C) logits are averaged over B·T
    regardless of whether they were flattened to 2-D first.
    """

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        self._probs = exp_x / exp_x.sum(axis=-1, keepdims=True)

        lead, n_classes = logits.shape[:-1], logits.shape[-1]
        target_shape = tuple(np.shape(targets))
        if target_shape == tuple(logits.shape):
            self._one_hot = targets
        elif target_shape == tuple(lead):
            # comparison-based one-hot: works on immutable (JAX) arrays too
            classes = np.arange(n_classes)
            self._one_hot = (
                targets.astype(int)[..., None] == classes
            ).astype(self._probs.dtype)
        else:
            raise ValueError(
                f"targets shape {target_shape} matches neither integer labels "
                f"{tuple(lead)} nor one-hot {tuple(logits.shape)} for logits of "
                f"shape {tuple(logits.shape)}"
            )

        self._n = max(int(onp.prod(lead)), 1)  # predicted positions, not batches
        return float(-np.sum(self._one_hot * np.log(self._probs + 1e-12)) / self._n)

    def backward(self) -> np.ndarray:
        return (self._probs - self._one_hot) / self._n


class BCEWithLogits(Loss):
    """Numerically stable binary cross-entropy on **raw logits**.

    `BinaryCrossEntropy` takes probabilities, so a saturated sigmoid upstream
    has already rounded to exactly 0 or 1 by the time the loss sees it, and the
    clip that prevents log(0) also flattens the gradient. Fusing the sigmoid in
    avoids that entirely — the same reason `SoftmaxCrossEntropy` is fused:

        loss = max(z, 0) − z·y + log(1 + exp(−|z|))

    which is algebraically identical to −[y·log σ(z) + (1−y)·log(1−σ(z))] but
    never exponentiates a positive number. The gradient is simply σ(z) − y.
    """

    def forward(self, logits: np.ndarray, target: np.ndarray) -> float:
        self._logits = logits
        self._target = target
        loss = (
            np.maximum(logits, 0.0)
            - logits * target
            + np.log1p(np.exp(-np.abs(logits)))
        )
        return float(np.mean(loss))

    def backward(self) -> np.ndarray:
        sigmoid = 1.0 / (1.0 + np.exp(-self._logits))
        return (sigmoid - self._target) / self._logits.size


class BinaryCrossEntropy(Loss):
    """Element-wise binary cross-entropy — expects sigmoid-activated predictions.

    Prefer `BCEWithLogits` when you control the model's last layer: it takes
    raw logits and is stable where this one has to clip.
    """

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self._pred = np.clip(pred, 1e-9, 1.0 - 1e-9)
        self._target = target
        return float(-np.mean(
            target * np.log(self._pred) + (1.0 - target) * np.log(1.0 - self._pred)
        ))

    def backward(self) -> np.ndarray:
        n = self._pred.size
        return (-(self._target / self._pred) + (1.0 - self._target) / (1.0 - self._pred)) / n
