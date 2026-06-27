"""
Activation functions with analytic backward passes.
"""
import numpy as np
from tiny_ml.core.module import Activation


class ReLU(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self._mask


class Sigmoid(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._out = 1.0 / (1.0 + np.exp(-x))
        return self._out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self._out * (1.0 - self._out)


class Tanh(Activation):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._out = np.tanh(x)
        return self._out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (1.0 - self._out ** 2)


class GeLU(Activation):
    """Gaussian Error Linear Unit — used in modern transformers."""
    _SQRT2 = np.sqrt(2.0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        from scipy.special import erf
        self._x = x
        self._cdf = 0.5 * (1.0 + erf(x / self._SQRT2))
        return x * self._cdf

    def backward(self, grad: np.ndarray) -> np.ndarray:
        pdf = np.exp(-0.5 * self._x ** 2) / np.sqrt(2.0 * np.pi)
        return grad * (self._cdf + self._x * pdf)
