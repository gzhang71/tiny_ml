import numpy as np
from core.module import Layer
from layers.linear import Linear
from layers.activations import ReLU


class ResidualBlock(Layer):
    """Two-layer residual block (MLP-style, no convolutions).

    out = ReLU(x + Linear2(ReLU(Linear1(x))))
    Requires in_features == out_features for the identity skip.
    """

    def __init__(self, features: int):
        self.linear1 = Linear(features, features)
        self.relu1 = ReLU()
        self.linear2 = Linear(features, features)
        self.relu2 = ReLU()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._skip = x
        out = self.relu1.forward(self.linear1.forward(x))
        out = out + x
        return self.relu2.forward(out)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad = self.relu2.backward(grad)
        grad_skip = grad
        grad = self.linear1.backward(self.relu1.backward(self.linear2.backward(grad)))
        return grad + grad_skip

    def parameters(self) -> list:
        return self.linear1.parameters() + self.linear2.parameters()
