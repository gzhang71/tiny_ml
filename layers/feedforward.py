import numpy as np
from core.module import Layer
from layers.linear import Linear
from layers.activations import ReLU


class FeedForward(Layer):
    """Position-wise FFN: Linear → activation → Linear (4× expansion).

    activation_cls defaults to ReLU; pass GeLU for GPT-2 style.
    """

    def __init__(self, d_model: int, d_ff: int | None = None, activation_cls=None):
        d_ff = d_ff or 4 * d_model
        self.linear1 = Linear(d_model, d_ff)
        self.act = (activation_cls or ReLU)()
        self.linear2 = Linear(d_ff, d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.linear2.forward(self.act.forward(self.linear1.forward(x)))

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return self.linear1.backward(self.act.backward(self.linear2.backward(grad)))

    def parameters(self) -> list:
        return self.linear1.parameters() + self.linear2.parameters()
