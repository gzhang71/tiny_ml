import numpy as np
from tiny_ml.core.module import Model, Layer


class Sequential(Model):
    """Ordered container of layers; runs forward/backward in sequence."""

    def __init__(self, layers: list[Layer]):
        self._layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self._layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self._layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self) -> list:
        params = []
        for layer in self._layers:
            params.extend(layer.parameters())
        return params
