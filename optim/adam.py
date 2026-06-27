import numpy as np
from tiny_ml.core.module import Optimizer
from tiny_ml.core.prameter import Parameter


class ADAM(Optimizer):
    """Adaptive Moment Estimation (Adam): m = β1*m + (1-β1)*g, v = β2*v + (1-β2)*g²"""

    def __init__(
        self,
        parameters: list[Parameter],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self._params = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m = [np.zeros_like(p.data) for p in parameters]
        self._v = [np.zeros_like(p.data) for p in parameters]
        self._t = 0

    def step(self):
        self._t += 1
        bc1 = 1.0 - self.beta1 ** self._t
        bc2 = 1.0 - self.beta2 ** self._t
        for i, p in enumerate(self._params):
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * p.grad
            self._v[i] = self.beta2 * self._v[i] + (1.0 - self.beta2) * p.grad ** 2
            m_hat = self._m[i] / bc1
            v_hat = self._v[i] / bc2
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self._params:
            p.grad = np.zeros_like(p.data)
