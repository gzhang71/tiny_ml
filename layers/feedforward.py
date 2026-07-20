from core.backend import xp as np
from core.module import Layer
from layers.linear import Linear
from layers.activations import ReLU, SiLU


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


class SwiGLU(Layer):
    """Gated FFN (Shazeer 2020, used by Llama): y = W2 (SiLU(W1 x) ⊙ W3 x).

    activation_cls defaults to SiLU; other activations give the sibling GLU
    variants (GeLU → GeGLU, Sigmoid → the original GLU). d_ff defaults to
    8/3 × d_model so the parameter count matches a 4× FeedForward despite
    the third projection.

    Same constructor signature as FeedForward, so it can be swapped into
    TransformerBlock via ffn_cls=SwiGLU.
    """

    def __init__(self, d_model: int, d_ff: int | None = None, activation_cls=None):
        d_ff = d_ff or (8 * d_model) // 3
        self.linear1 = Linear(d_model, d_ff)   # gate branch (activated)
        self.linear3 = Linear(d_model, d_ff)   # value branch (linear)
        self.act = (activation_cls or SiLU)()
        self.linear2 = Linear(d_ff, d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._gate = self.act.forward(self.linear1.forward(x))
        self._value = self.linear3.forward(x)
        return self.linear2.forward(self._gate * self._value)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        d_h = self.linear2.backward(grad)
        d_gate = self.linear1.backward(self.act.backward(d_h * self._value))
        d_value = self.linear3.backward(d_h * self._gate)
        return d_gate + d_value

    def parameters(self) -> list:
        return (
            self.linear1.parameters() + self.linear3.parameters()
            + self.linear2.parameters()
        )
