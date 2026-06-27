"""
GPT-style (decoder-only) Transformer built entirely from numpy.

Components exposed:
  LayerNorm, MultiHeadAttention, FeedForward, TransformerBlock, Transformer
"""
import numpy as np
from tiny_ml.core.module import Layer, Model
from tiny_ml.core.prameter import Parameter
from tiny_ml.layers.linear import Linear


# ---------------------------------------------------------------------------
# Layer Norm
# ---------------------------------------------------------------------------

class LayerNorm(Layer):
    """Normalises the last dimension: y = (x - μ) / σ * γ + β"""

    def __init__(self, d_model: int, eps: float = 1e-5):
        self.gamma = Parameter(np.ones(d_model))
        self.beta = Parameter(np.zeros(d_model))
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        self._std_inv = 1.0 / np.sqrt(var + self.eps)
        self._x_norm = (x - mu) * self._std_inv
        return self.gamma.data * self._x_norm + self.beta.data

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # accumulate parameter grads
        reduce_axes = tuple(range(grad.ndim - 1))
        self.gamma.grad += (grad * self._x_norm).sum(axis=reduce_axes)
        self.beta.grad += grad.sum(axis=reduce_axes)

        # gradient through normalisation (Ba et al. 2016 eq.)
        d_xn = grad * self.gamma.data
        N = grad.shape[-1]
        dx = self._std_inv * (
            N * d_xn
            - d_xn.sum(axis=-1, keepdims=True)
            - self._x_norm * (d_xn * self._x_norm).sum(axis=-1, keepdims=True)
        ) / N
        return dx


# ---------------------------------------------------------------------------
# Embedding (token + positional)
# ---------------------------------------------------------------------------

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


def _sinusoidal_pe(seq_len: int, d_model: int) -> np.ndarray:
    pos = np.arange(seq_len)[:, None]
    div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe[None]  # (1, seq, d_model)


# ---------------------------------------------------------------------------
# Multi-Head (Causal) Self-Attention
# ---------------------------------------------------------------------------

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

    # -- helpers --

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        # (B, T, d_model) → (B, n_heads, T, d_k)
        B, T, _ = x.shape
        x = x.reshape(B, T, self.n_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        # (B, n_heads, T, d_k) → (B, T, d_model)
        B, H, T, dk = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * dk)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)

    @staticmethod
    def _softmax_backward(grad: np.ndarray, s: np.ndarray) -> np.ndarray:
        # s * (grad - (grad * s).sum(-1, keepdims=True))
        return s * (grad - (grad * s).sum(axis=-1, keepdims=True))

    # -- forward / backward --

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, T, _ = x.shape

        Q = self._split_heads(self.W_Q.forward(x))  # (B, H, T, dk)
        K = self._split_heads(self.W_K.forward(x))
        V = self._split_heads(self.W_V.forward(x))

        scale = np.sqrt(self.d_k)
        scores = Q @ K.transpose(0, 1, 3, 2) / scale  # (B, H, T, T)

        if self.causal:
            mask = np.triu(np.ones((T, T), dtype=bool), k=1)
            scores = np.where(mask, -1e9, scores)

        attn = self._softmax(scores)  # (B, H, T, T)

        # cache for backward
        self._Q, self._K, self._V = Q, K, V
        self._attn = attn
        self._scale = scale
        self._x = x
        if self.causal:
            self._causal_mask = mask

        out = attn @ V                            # (B, H, T, dk)
        out = self._merge_heads(out)              # (B, T, d_model)
        return self.W_O.forward(out)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        B, T, _ = grad.shape

        d_merged = self.W_O.backward(grad)                # (B, T, d_model)
        d_out = self._split_heads(d_merged)               # (B, H, T, dk)

        d_V = self._attn.transpose(0, 1, 3, 2) @ d_out   # (B, H, T, dk)
        d_attn = d_out @ self._V.transpose(0, 1, 3, 2)   # (B, H, T, T)

        d_scores = self._softmax_backward(d_attn, self._attn)
        if self.causal:
            d_scores = np.where(self._causal_mask, 0.0, d_scores)
        d_scores /= self._scale

        d_Q = d_scores @ self._K                          # (B, H, T, dk)
        d_K = d_scores.transpose(0, 1, 3, 2) @ self._Q   # (B, H, T, dk)

        d_q = self.W_Q.backward(self._merge_heads(d_Q))
        d_k = self.W_K.backward(self._merge_heads(d_K))
        d_v = self.W_V.backward(self._merge_heads(d_V))
        return d_q + d_k + d_v

    def parameters(self) -> list:
        return (
            self.W_Q.parameters()
            + self.W_K.parameters()
            + self.W_V.parameters()
            + self.W_O.parameters()
        )


# ---------------------------------------------------------------------------
# Feed-Forward sub-layer
# ---------------------------------------------------------------------------

class FeedForward(Layer):
    """Position-wise FFN: Linear → ReLU → Linear (4× expansion)."""

    def __init__(self, d_model: int, d_ff: int | None = None):
        d_ff = d_ff or 4 * d_model
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._pre_act = self.linear1.forward(x)
        self._act_mask = self._pre_act > 0          # ReLU mask
        act = self._pre_act * self._act_mask
        return self.linear2.forward(act)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad = self.linear2.backward(grad)
        grad = grad * self._act_mask                # ReLU backward
        return self.linear1.backward(grad)

    def parameters(self) -> list:
        return self.linear1.parameters() + self.linear2.parameters()


# ---------------------------------------------------------------------------
# Transformer Block (Pre-LN, GPT-2 style)
# ---------------------------------------------------------------------------

class TransformerBlock(Layer):
    """
    Pre-norm residual block:
      x = x + Attention(LayerNorm1(x))
      x = x + FFN(LayerNorm2(x))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int | None = None, causal: bool = True):
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, causal=causal)
        self.norm2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # attention sub-layer
        self._x1 = x
        normed1 = self.norm1.forward(x)
        attn_out = self.attn.forward(normed1)
        x = x + attn_out

        # FFN sub-layer
        self._x2 = x
        normed2 = self.norm2.forward(x)
        ffn_out = self.ffn.forward(normed2)
        return x + ffn_out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # FFN residual branch
        d_ffn = self.ffn.backward(self.norm2.backward(grad))
        grad = grad + d_ffn                 # add grad from identity + FFN path

        # Attention residual branch
        d_attn = self.attn.backward(self.norm1.backward(grad))
        return grad + d_attn

    def parameters(self) -> list:
        return (
            self.norm1.parameters()
            + self.attn.parameters()
            + self.norm2.parameters()
            + self.ffn.parameters()
        )


# ---------------------------------------------------------------------------
# Transformer (GPT-style decoder)
# ---------------------------------------------------------------------------

class Transformer(Model):
    """
    GPT-style decoder-only transformer.

    forward(tokens) → logits of shape (B, T, vocab_size)

    tokens: integer array of shape (B, T)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_seq_len: int = 512,
        d_ff: int | None = None,
    ):
        self.token_emb = Embedding(vocab_size, d_model)
        self._pe = _sinusoidal_pe(max_seq_len, d_model)  # (1, max_seq, d_model)
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff, causal=True)
            for _ in range(n_layers)
        ]
        self.norm = LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size)

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        T = tokens.shape[1]
        x = self.token_emb.forward(tokens) + self._pe[:, :T]
        self._emb_out = x
        for block in self.blocks:
            x = block.forward(x)
        x = self.norm.forward(x)
        return self.head.forward(x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad = self.head.backward(grad)
        grad = self.norm.backward(grad)
        for block in reversed(self.blocks):
            grad = block.backward(grad)
        self.token_emb.backward(grad)   # scatter-adds into embedding grad
        return None

    def parameters(self) -> list:
        params = self.token_emb.parameters()
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.norm.parameters())
        params.extend(self.head.parameters())
        return params
