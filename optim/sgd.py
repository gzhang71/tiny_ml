import numpy as np
from tiny_ml.core.module import Optimizer
from tiny_ml.core.prameter import Parameter


class SGD(Optimizer):
    def __init__(self, parameters: list[Parameter], lr: float = 0.01):
        self._params = parameters
        self.lr = lr

    def step(self):
        for p in self._params:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self._params:
            p.grad = np.zeros_like(p.data)
