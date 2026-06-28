import numpy as np
from tiny_ml.core.module import Layer
from tiny_ml.core.prameter import Parameter


class LayerNorm(Layer):
    """Normalises the last dimension: y = (x - μ) / σ * γ + β"""

    def __init__(self, d_model: int, eps: float = 1e-5):
        self.gamma = Parameter(np.ones(d_model))
        self.beta = Parameter(np.zeros(d_model))
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        self._std_inv = 1.0 / np.sqrt(var + self.eps)
        self._x_norm = (x - mu) * self._std_inv
        return self.gamma.data * self._x_norm + self.beta.data

    def backward(self, grad: np.ndarray) -> np.ndarray:
        reduce_axes = tuple(range(grad.ndim - 1))
        self.gamma.grad += (grad * self._x_norm).sum(axis=reduce_axes)
        self.beta.grad += grad.sum(axis=reduce_axes)

        d_xn = grad * self.gamma.data
        N = grad.shape[-1]
        return self._std_inv * (
            N * d_xn
            - d_xn.sum(axis=-1, keepdims=True)
            - self._x_norm * (d_xn * self._x_norm).sum(axis=-1, keepdims=True)
        ) / N
