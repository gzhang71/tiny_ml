import numpy as np
from core.module import Layer
from core.parameter import Parameter
from layers.linear import Linear
from layers.normalization import LayerNorm
from layers.feedforward import FeedForward
from layers.embedding import Embedding


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _softmax_backward(grad: np.ndarray, s: np.ndarray) -> np.ndarray:
    return s * (grad - (grad * s).sum(axis=-1, keepdims=True))


class MultiHeadAttention(Layer):
    """Scaled dot-product multi-head self-attention with optional causal mask."""

    def __init__(self, d_model: int, n_heads: int, causal: bool = True):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.causal = causal
        self.W_Q = Linear(d_model, d_model)
        self.W_K = Linear(d_model, d_model)
        self.W_V = Linear(d_model, d_model)
        self.W_O = Linear(d_model, d_model)

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        B, T, _ = x.shape
        return x.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        B, H, T, dk = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * dk)

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, T, _ = x.shape
        Q = self._split_heads(self.W_Q.forward(x))
        K = self._split_heads(self.W_K.forward(x))
        V = self._split_heads(self.W_V.forward(x))

        scale = np.sqrt(self.d_k)
        scores = Q @ K.transpose(0, 1, 3, 2) / scale

        if self.causal:
            mask = np.triu(np.ones((T, T), dtype=bool), k=1)
            scores = np.where(mask, -1e9, scores)
            self._causal_mask = mask

        attn = _softmax(scores)
        self._Q, self._K, self._V, self._attn, self._scale = Q, K, V, attn, scale
        return self.W_O.forward(self._merge_heads(attn @ V))

    def backward(self, grad: np.ndarray) -> np.ndarray:
        d_out = self._split_heads(self.W_O.backward(grad))

        d_V = self._attn.transpose(0, 1, 3, 2) @ d_out
        d_attn = d_out @ self._V.transpose(0, 1, 3, 2)

        d_scores = _softmax_backward(d_attn, self._attn)
        if self.causal:
            d_scores = np.where(self._causal_mask, 0.0, d_scores)
        d_scores /= self._scale

        d_Q = d_scores @ self._K
        d_K = d_scores.transpose(0, 1, 3, 2) @ self._Q
        return (
            self.W_Q.backward(self._merge_heads(d_Q))
            + self.W_K.backward(self._merge_heads(d_K))
            + self.W_V.backward(self._merge_heads(d_V))
        )

    def parameters(self) -> list:
        return (
            self.W_Q.parameters() + self.W_K.parameters()
            + self.W_V.parameters() + self.W_O.parameters()
        )


class TransformerBlock(Layer):
    """Pre-norm residual block: x = x + Attn(LN(x));  x = x + FFN(LN(x))."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int | None = None,
        causal: bool = True,
        activation_cls=None,
    ):
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, causal=causal)
        self.norm2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, activation_cls=activation_cls)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x + self.attn.forward(self.norm1.forward(x))
        x = x + self.ffn.forward(self.norm2.forward(x))
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad = grad + self.norm2.backward(self.ffn.backward(grad))
        grad = grad + self.norm1.backward(self.attn.backward(grad))
        return grad

    def parameters(self) -> list:
        return (
            self.norm1.parameters() + self.attn.parameters()
            + self.norm2.parameters() + self.ffn.parameters()
        )


class RelativePositionBias(Layer):
    """Learnable scalar bias per (head, relative-position-bucket) pair (T5 style).

    forward(seq_q, seq_k) → (n_heads, seq_q, seq_k) to be added to scores.
    """

    def __init__(
        self,
        n_heads: int,
        n_buckets: int = 32,
        max_distance: int = 128,
        bidirectional: bool = True,
    ):
        self.n_heads = n_heads
        self.n_buckets = n_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        self.W = Parameter(np.zeros((n_heads, n_buckets)))

    def _bucket(self, rel: np.ndarray) -> np.ndarray:
        n = self.n_buckets
        ret = np.zeros_like(rel, dtype=int)
        if self.bidirectional:
            n //= 2
            ret += (rel > 0).astype(int) * n
            rel = np.abs(rel)
        else:
            rel = -np.minimum(rel, 0)
        max_exact = n // 2
        is_small = rel < max_exact
        val_large = (
            max_exact
            + (
                np.log(np.maximum(rel.astype(float), 1.0) / max_exact)
                / np.log(self.max_distance / max_exact)
                * (n - max_exact)
            ).astype(int)
        )
        return (ret + np.where(is_small, rel, np.minimum(val_large, n - 1))).astype(int)

    def forward(self, seq_q: int, seq_k: int) -> np.ndarray:
        q_pos = np.arange(seq_q)[:, None]
        k_pos = np.arange(seq_k)[None, :]
        self._buckets = self._bucket(q_pos - k_pos)
        return self.W.data[:, self._buckets]

    def backward(self, grad: np.ndarray) -> None:
        for h in range(self.n_heads):
            np.add.at(self.W.grad[h], self._buckets, grad[h])


class T5SelfAttention(Layer):
    """Multi-head self-attention with relative position bias."""

    def __init__(self, d_model: int, n_heads: int, causal: bool, n_buckets: int = 32):
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.causal = causal
        self.W_Q = Linear(d_model, d_model)
        self.W_K = Linear(d_model, d_model)
        self.W_V = Linear(d_model, d_model)
        self.W_O = Linear(d_model, d_model)
        self.rel_bias = RelativePositionBias(n_heads, n_buckets, bidirectional=not causal)

    def _split(self, x: np.ndarray) -> np.ndarray:
        B, T, _ = x.shape
        return x.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

    def _merge(self, x: np.ndarray) -> np.ndarray:
        B, H, T, dk = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * dk)

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, T, _ = x.shape
        Q = self._split(self.W_Q.forward(x))
        K = self._split(self.W_K.forward(x))
        V = self._split(self.W_V.forward(x))

        scale = np.sqrt(self.d_k)
        scores = Q @ K.transpose(0, 1, 3, 2) / scale
        scores = scores + self.rel_bias.forward(T, T)[None]

        if self.causal:
            mask = np.triu(np.ones((T, T), dtype=bool), k=1)
            scores = np.where(mask, -1e9, scores)
            self._causal_mask = mask

        attn = _softmax(scores)
        self._Q, self._K, self._V, self._attn, self._scale = Q, K, V, attn, scale
        return self.W_O.forward(self._merge(attn @ V))

    def backward(self, grad: np.ndarray) -> np.ndarray:
        d_out = self._split(self.W_O.backward(grad))

        d_V = self._attn.transpose(0, 1, 3, 2) @ d_out
        d_attn = d_out @ self._V.transpose(0, 1, 3, 2)

        d_scores = _softmax_backward(d_attn, self._attn)
        if self.causal:
            d_scores = np.where(self._causal_mask, 0.0, d_scores)

        self.rel_bias.backward(d_scores.sum(axis=0))
        d_scores = d_scores / self._scale

        d_Q = d_scores @ self._K
        d_K = d_scores.transpose(0, 1, 3, 2) @ self._Q
        return (
            self.W_Q.backward(self._merge(d_Q))
            + self.W_K.backward(self._merge(d_K))
            + self.W_V.backward(self._merge(d_V))
        )

    def parameters(self) -> list:
        return (
            self.W_Q.parameters() + self.W_K.parameters()
            + self.W_V.parameters() + self.W_O.parameters()
            + self.rel_bias.parameters()
        )


class CrossAttention(Layer):
    """Multi-head cross-attention: Q ← decoder, K/V ← encoder."""

    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_Q = Linear(d_model, d_model)
        self.W_K = Linear(d_model, d_model)
        self.W_V = Linear(d_model, d_model)
        self.W_O = Linear(d_model, d_model)

    def _split(self, x: np.ndarray) -> np.ndarray:
        B, T, _ = x.shape
        return x.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

    def _merge(self, x: np.ndarray) -> np.ndarray:
        B, H, T, dk = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * dk)

    def forward(self, x_dec: np.ndarray, x_enc: np.ndarray) -> np.ndarray:
        scale = np.sqrt(self.d_k)
        Q = self._split(self.W_Q.forward(x_dec))
        K = self._split(self.W_K.forward(x_enc))
        V = self._split(self.W_V.forward(x_enc))

        attn = _softmax(Q @ K.transpose(0, 1, 3, 2) / scale)
        self._Q, self._K, self._V, self._attn, self._scale = Q, K, V, attn, scale
        return self.W_O.forward(self._merge(attn @ V))

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (d_x_dec, d_x_enc)."""
        d_out = self._split(self.W_O.backward(grad))

        d_V = self._attn.transpose(0, 1, 3, 2) @ d_out
        d_attn = d_out @ self._V.transpose(0, 1, 3, 2)
        d_scores = _softmax_backward(d_attn, self._attn) / self._scale

        d_Q = d_scores @ self._K
        d_K = d_scores.transpose(0, 1, 3, 2) @ self._Q
        return (
            self.W_Q.backward(self._merge(d_Q)),
            self.W_K.backward(self._merge(d_K)) + self.W_V.backward(self._merge(d_V)),
        )

    def parameters(self) -> list:
        return (
            self.W_Q.parameters() + self.W_K.parameters()
            + self.W_V.parameters() + self.W_O.parameters()
        )


class _TiedProjection(Layer):
    """Output head sharing the token embedding matrix (weight tying)."""

    def __init__(self, embedding: Embedding):
        self._W = embedding.W

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self._W.data.T

    def backward(self, grad: np.ndarray) -> np.ndarray:
        x_2d = self._x.reshape(-1, self._x.shape[-1])
        grad_2d = grad.reshape(-1, grad.shape[-1])
        self._W.grad += grad_2d.T @ x_2d
        return (grad_2d @ self._W.data).reshape(self._x.shape)

    def parameters(self) -> list:
        return []
