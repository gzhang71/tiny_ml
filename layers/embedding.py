import numpy as np
from tiny_ml.core.module import Layer
from tiny_ml.core.prameter import Parameter


class Embedding(Layer):
    """Integer token → dense vector lookup."""

    def __init__(self, vocab_size: int, d_model: int):
        self.W = Parameter(np.random.randn(vocab_size, d_model) * 0.02)
        self._tokens: np.ndarray | None = None

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        self._tokens = tokens
        return self.W.data[tokens]

    def backward(self, grad: np.ndarray) -> np.ndarray:
        np.add.at(self.W.grad, self._tokens, grad)
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
        pe = np.zeros((max_seq_len, d_model))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div)
        self._pe = pe[None]  # (1, max_seq_len, d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        T = x.shape[1]
        return x + self._pe[:, :T]

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
        self.W = Parameter(np.random.randn(max_seq_len, d_model) * 0.02)
        self._T: int = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._T = x.shape[1]
        return x + self.W.data[: self._T]

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.W.grad[: self._T] += grad.sum(axis=0)  # sum over batch
        return grad  # identity for the residual path


class FeatureEmbedding(Layer):
    """Projects each continuous input feature independently into d_model dims.

    Useful for tabular transformers where each column becomes a "token".

    forward(x: (B, n_features)) → (B, n_features, d_model)

    Each feature i has its own weight vector W[i] of size d_model:
        out[b, i, :] = x[b, i] * W[i] + bias[i]
    """

    def __init__(self, n_features: int, d_model: int):
        self.W = Parameter(np.random.randn(n_features, d_model) * np.sqrt(2.0 / n_features))
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
