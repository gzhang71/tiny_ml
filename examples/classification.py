"""
Classification demo: 2-class spiral dataset classified with an MLP.
Run from the repo root: python -m tiny_ml.examples.classification
"""
import numpy as np
from tiny_ml.models.mlp import MLP
from tiny_ml.losses.losses import SoftmaxCrossEntropy
from tiny_ml.optim.adam import ADAM
from tiny_ml.training.trainer import Trainer


def make_spiral(n_per_class: int = 200, noise: float = 0.15) -> tuple:
    """Two interleaved spirals."""
    X, y = [], []
    for c in range(2):
        angle_offset = c * np.pi
        t = np.linspace(0, 2 * np.pi, n_per_class)
        r = t / (2 * np.pi)
        x1 = r * np.cos(t + angle_offset) + noise * np.random.randn(n_per_class)
        x2 = r * np.sin(t + angle_offset) + noise * np.random.randn(n_per_class)
        X.append(np.stack([x1, x2], axis=1))
        y.append(np.full(n_per_class, c, dtype=int))
    return np.concatenate(X).astype(np.float32), np.concatenate(y)


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return float((logits.argmax(axis=1) == labels).mean())


def main():
    np.random.seed(42)

    x, y = make_spiral(n_per_class=300)

    model = MLP([2, 128, 64, 2])
    loss_fn = SoftmaxCrossEntropy()
    optimizer = ADAM(model.parameters(), lr=3e-3)

    trainer = Trainer(model, loss_fn, optimizer)
    trainer.fit(x, y, epochs=300, batch_size=64)

    logits = trainer.predict(x)
    print(f"\nTraining accuracy: {accuracy(logits, y) * 100:.1f}%")


if __name__ == "__main__":
    main()
