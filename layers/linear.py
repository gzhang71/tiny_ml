import numpy as np
from tiny_ml.core.module import Layer
from tiny_ml.core.prameter import Parameter


class Linear(Layer):
    """Fully-connected layer: out = x @ W + b"""

    def __init__(self, in_features: int, out_features: int):
        self.W = Parameter(np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features))
        self.b = Parameter(np.zeros(out_features))
        self._input: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        return x @ self.W.data + self.b.data

    def backward(self, grad: np.ndarray) -> np.ndarray:
        x = self._input
        original_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        grad_2d = grad.reshape(-1, grad.shape[-1])

        self.W.grad += x_2d.T @ grad_2d
        self.b.grad += grad_2d.sum(axis=0)
        return (grad_2d @ self.W.data.T).reshape(original_shape)
