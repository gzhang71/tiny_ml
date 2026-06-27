"""
Abstract base classes for all modules.
"""
import numpy as np
from abc import ABC, abstractmethod
from tiny_ml.core.prameter import Parameter


class Module(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray: ...

    def parameters(self) -> list:
        params = []
        for attr in self.__dict__.values():
            if isinstance(attr, Parameter):
                params.append(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Layer(Module):
    pass


class Activation(Layer):
    pass


class Loss(ABC):
    @abstractmethod
    def forward(self, pred: np.ndarray, target: np.ndarray) -> float: ...

    @abstractmethod
    def backward(self) -> np.ndarray: ...

    def __call__(self, pred, target):
        return self.forward(pred, target)


class Optimizer(ABC):
    @abstractmethod
    def step(self): ...

    @abstractmethod
    def zero_grad(self): ...


class Model(Module):
    pass
