import numpy as np
from core.module import Model
from layers.linear import Linear
from layers.activations import ReLU
from layers.residual import ResidualBlock


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
        x = self.input_relu.forward(self.input_proj.forward(x))
        for block in self.blocks:
            x = block.forward(x)
        return self.output_proj.forward(x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad = self.output_proj.backward(grad)
        for block in reversed(self.blocks):
            grad = block.backward(grad)
        return self.input_proj.backward(self.input_relu.backward(grad))

    def parameters(self) -> list:
        params = self.input_proj.parameters()
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.output_proj.parameters())
        return params
