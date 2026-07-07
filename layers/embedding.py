from core.backend import xp as np, randn, scatter_add, take_slice
from core.module import Layer
from core.parameter import Parameter


class Embedding(Layer):
    """Integer token → dense vector lookup."""

    def __init__(self, vocab_size: int, d_model: int):
        self.W = Parameter(randn(vocab_size, d_model) * 0.02)
        self._tokens: np.ndarray | None = None

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        self._tokens = tokens
        return self.W.data[tokens]

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.W.grad = scatter_add(self.W.grad, self._tokens, grad)
        return None  # no gradient flows to integer token indices


class SinusoidalPositionalEmbedding(Layer):
    """Fixed sine/cosine positional encoding (Vaswani et al. 2017).

    Has no learnable parameters; forward returns the PE table slice to be
    added to token embeddings.

    forward(x: (B, T, d_model)) → (B, T, d_model)
    """

    def __init__(self, d_model: int, max_seq_len: int = 512):
        pos = np.arange(max_seq_len)[:, None]
        div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        # interleave sin (even dims) and cos (odd dims) without item assignment
        pe = np.stack([np.sin(pos * div), np.cos(pos * div)], axis=2).reshape(max_seq_len, d_model)
        self._pe = pe[None]  # (1, max_seq_len, d_model)

    def forward(self, x: np.ndarray, offset: int = 0) -> np.ndarray:
        T = x.shape[1]
        # take_slice: offset changes every decode step; a baked-in slice would
        # recompile per step in jax mode
        return x + take_slice(self._pe, offset, T, axis=1)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad  # PE is constant; gradient passes through unchanged

    def parameters(self) -> list:
        return []


class LearnedPositionalEmbedding(Layer):
    """Learnable position embedding (GPT-2 style).

    Looks up a position vector for each timestep and adds it to the input.

    forward(x: (B, T, d_model)) → (B, T, d_model)
    """

    def __init__(self, max_seq_len: int, d_model: int):
        self.W = Parameter(randn(max_seq_len, d_model) * 0.02)
        self._T: int = 0
        self._offset: int = 0

    def forward(self, x: np.ndarray, offset: int = 0) -> np.ndarray:
        self._T = x.shape[1]
        self._offset = offset
        return x + take_slice(self.W.data, offset, self._T, axis=0)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.W.grad = scatter_add(
            self.W.grad, slice(self._offset, self._offset + self._T), grad.sum(axis=0)
        )  # sum over batch
        return grad  # identity for the residual path


class FeatureEmbedding(Layer):
    """Projects each continuous input feature independently into d_model dims.

    Useful for tabular transformers where each column becomes a "token".

    forward(x: (B, n_features)) → (B, n_features, d_model)

    Each feature i has its own weight vector W[i] of size d_model:
        out[b, i, :] = x[b, i] * W[i] + bias[i]
    """

    def __init__(self, n_features: int, d_model: int):
        self.W = Parameter(randn(n_features, d_model) * np.sqrt(2.0 / n_features))
        self.bias = Parameter(np.zeros((n_features, d_model)))
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: (B, n_features)
        self._x = x
        # broadcast: (B, n_features, 1) * (n_features, d_model) → (B, n_features, d_model)
        return x[:, :, None] * self.W.data[None] + self.bias.data[None]

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # grad: (B, n_features, d_model)
        self.W.grad += (self._x[:, :, None] * grad).sum(axis=0)
        self.bias.grad += grad.sum(axis=0)
        # d_x[b, i] = sum_d(W[i, d] * grad[b, i, d])
        return (self.W.data[None] * grad).sum(axis=-1)  # (B, n_features)
