import numpy as np
from tiny_ml.core.module import Optimizer
from tiny_ml.core.prameter import Parameter


class Momentum(Optimizer):
    """SGD with heavy-ball momentum: v = μv + g,  θ = θ - lr * v"""

    def __init__(self, parameters: list[Parameter], lr: float = 0.01, momentum: float = 0.9):
        self._params = parameters
        self.lr = lr
        self.momentum = momentum
        self._velocity = [np.zeros_like(p.data) for p in parameters]

    def step(self):
        for i, p in enumerate(self._params):
            self._velocity[i] = self.momentum * self._velocity[i] + p.grad
            p.data -= self.lr * self._velocity[i]

    def zero_grad(self):
        for p in self._params:
            p.grad = np.zeros_like(p.data)
