"""
Head comparison example: the same MLP feature body topped with each
classification-head geometry — affine (LinearHead), Euclidean prototypes
(EuclideanHead), cosine similarity (CosineHead), and Poincaré-ball geodesic
distance (HyperbolicHead) — trained on the 2-class spiral.
Run: python -m examples.heads
"""
import numpy as np
from core.module import Model
from core.backend import to_numpy
from models.mlp import MLP
from layers.heads import LinearHead, EuclideanHead, CosineHead, HyperbolicHead
from losses.losses import SoftmaxCrossEntropy
from optim.adam import ADAM
from training.trainer import Trainer
from examples.mlp import make_spiral


class HeadedMLP(Model):
    """MLP feature body with a swappable classification head."""

    def __init__(self, head_cls, d_feature: int = 8, n_classes: int = 2):
        self.body = MLP([2, 64, 32, d_feature])
        self.head = head_cls(d_feature, n_classes)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.head.forward(self.body.forward(x))

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return self.body.backward(self.head.backward(grad))


def main():
    for head_cls in (LinearHead, EuclideanHead, CosineHead, HyperbolicHead):
        np.random.seed(42)
        x, y = make_spiral()

        model = HeadedMLP(head_cls)
        trainer = Trainer(model, SoftmaxCrossEntropy(), ADAM(model.parameters(), lr=3e-3))
        trainer.fit(x, y, epochs=100, batch_size=64, verbose=False)

        logits = to_numpy(trainer.predict(x))
        acc = (logits.argmax(axis=1) == y).mean()
        print(f"{head_cls.__name__:16s} accuracy: {acc * 100:.1f}%")


if __name__ == "__main__":
    main()
