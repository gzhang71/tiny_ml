from core.backend import xp as np, rand
from core.module import Layer


class Dropout(Layer):
    """Inverted dropout: zero each activation with probability p while training.

    Kept units are scaled by 1/(1−p) *during training*, so the expected value
    of the output matches the input and inference needs no rescaling at all —
    `forward` is the identity once the module is in eval mode. This is why the
    layer has to know which mode it is in, and why `Module.train()/eval()`
    exists: leaving dropout on at inference makes predictions randomly wrong,
    and rescaling at the wrong time silently shifts every activation.

    The mask is drawn fresh per forward and saved for backward, matching the
    layer contract (one backward per forward).

        block.ffn = Sequential([FeedForward(d_model), Dropout(0.1)])

    Note `p=0` short-circuits entirely, so a model built with dropout disabled
    pays nothing.
    """

    def __init__(self, p: float = 0.1):
        assert 0.0 <= p < 1.0, "dropout probability must be in [0, 1)"
        self.p = p
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0.0:
            self._mask = None
            return x
        keep = 1.0 - self.p
        # inverted dropout: scale at train time so eval is a plain identity
        self._mask = (rand(*x.shape) >= self.p) / keep
        return x * self._mask

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self._mask is None:
            return grad
        return grad * self._mask

    def parameters(self) -> list:
        return []
