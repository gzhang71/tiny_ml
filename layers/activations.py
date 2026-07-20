"""
Activation functions with analytic backward passes.
"""
from core.backend import xp as np
from core.module import Activation


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


class SiLU(Activation):
    """SiLU / Swish: x * sigmoid(x)."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        self._sig = 1.0 / (1.0 + np.exp(-x))
        return x * self._sig

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self._sig * (1.0 + self._x * (1.0 - self._sig))


class GeLU(Activation):
    """Tanh-approximated GeLU (used by GPT-2).

    GeLU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    """
    _C = np.sqrt(2.0 / np.pi)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        self._inner = self._C * (x + 0.044715 * x ** 3)
        self._tanh = np.tanh(self._inner)
        return 0.5 * x * (1.0 + self._tanh)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        cdf = 0.5 * (1.0 + self._tanh)
        d_tanh = 1.0 - self._tanh ** 2
        d_inner = self._C * (1.0 + 3.0 * 0.044715 * self._x ** 2)
        return grad * (cdf + self._x * 0.5 * d_tanh * d_inner)
