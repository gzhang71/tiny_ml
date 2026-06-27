import numpy as np
from tiny_ml.core.module import Model, Activation
from tiny_ml.layers.linear import Linear
from tiny_ml.layers.activations import ReLU
from tiny_ml.models.sequential import Sequential


class MLP(Model):
    """Multi-layer perceptron built from Linear + Activation stacks.

    layer_sizes: e.g. [784, 256, 128, 10]
    The activation is applied after every hidden layer but not the output layer.
    """

    def __init__(self, layer_sizes: list[int], activation: type[Activation] = ReLU):
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(activation())
        self._seq = Sequential(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._seq.forward(x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return self._seq.backward(grad)

    def parameters(self) -> list:
        return self._seq.parameters()
