import numpy as np
from tiny_ml.core.module import Model, Layer
from tiny_ml.core.prameter import Parameter
from tiny_ml.layers.linear import Linear
from tiny_ml.layers.activations import ReLU


class ResidualBlock(Layer):
    """Two-layer residual block (MLP-style, no convolutions).

    out = ReLU(x + Linear2(ReLU(Linear1(x))))
    Requires in_features == out_features for the identity skip.
    """

    def __init__(self, features: int):
        self.linear1 = Linear(features, features)
        self.relu1 = ReLU()
        self.linear2 = Linear(features, features)
        self.relu2 = ReLU()

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._skip = x
        out = self.linear1.forward(x)
        out = self.relu1.forward(out)
        out = self.linear2.forward(out)
        out = out + x                   # residual addition
        self._pre_relu2 = out
        return self.relu2.forward(out)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad = self.relu2.backward(grad)
        grad_skip = grad               # identity branch carries the same gradient
        grad = self.linear2.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.linear1.backward(grad)
        return grad + grad_skip

    def parameters(self) -> list:
        return self.linear1.parameters() + self.linear2.parameters()


class ResNet(Model):
    """Stack of residual blocks with input/output projection layers.

    Architecture: Linear_in → [ResidualBlock] x n → Linear_out
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_blocks: int = 2,
    ):
        self.input_proj = Linear(in_features, hidden_features)
        self.input_relu = ReLU()
        self.blocks = [ResidualBlock(hidden_features) for _ in range(num_blocks)]
        self.output_proj = Linear(hidden_features, out_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.input_proj.forward(x)
        x = self.input_relu.forward(x)
        for block in self.blocks:
            x = block.forward(x)
        return self.output_proj.forward(x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad = self.output_proj.backward(grad)
        for block in reversed(self.blocks):
            grad = block.backward(grad)
        grad = self.input_relu.backward(grad)
        return self.input_proj.backward(grad)

    def parameters(self) -> list:
        params = self.input_proj.parameters() + self.input_relu.parameters()
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.output_proj.parameters())
        return params
