"""
Abstract base classes for all modules.
"""
from core.backend import xp as np
from abc import ABC, abstractmethod
from core.parameter import Parameter


class Module(ABC):
    # Train/eval mode. Layers whose behavior differs between training and
    # inference (Dropout today) read `self.training`; everything else ignores
    # it. Class-level default so existing layers need no constructor change.
    training: bool = True

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray: ...

    def modules(self) -> list:
        """This module and every nested Module, depth-first.

        Walks `__dict__` the same way `parameters()` does. Note it cannot go
        through `parameters()` itself: layers like `Attention` override that
        method to return a hand-built list, and `_TiedProjection` deliberately
        returns nothing at all.
        """
        found = [self]
        for attr in self.__dict__.values():
            if isinstance(attr, Module):
                found.extend(attr.modules())
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, Module):
                        found.extend(item.modules())
        return found

    def train(self, mode: bool = True):
        """Put this module and all children in training mode. Returns self."""
        for module in self.modules():
            module.training = mode
        return self

    def eval(self):
        """Put this module and all children in inference mode. Returns self."""
        return self.train(False)

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

    def parameters(self) -> list:
        """The Parameter list this optimizer updates.

        Every optimizer stores it as `_params`; exposing it lets callers such
        as gradient clipping act on exactly the set being optimized.
        """
        return self._params


class Model(Module):
    pass
