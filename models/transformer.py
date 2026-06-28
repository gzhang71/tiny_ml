"""
GPT-style (decoder-only) Transformer built entirely from numpy.

Components exposed:
  MultiHeadAttention, FeedForward, TransformerBlock, Transformer
"""
import numpy as np
from tiny_ml.core.module import Layer, Model
from tiny_ml.layers.linear import Linear
from tiny_ml.layers.activations import ReLU
from tiny_ml.layers.normalization import LayerNorm
from tiny_ml.layers.embedding import Embedding, SinusoidalPositionalEmbedding


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

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        B, T, _ = x.shape
        return x.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        B, H, T, dk = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * dk)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)

    @staticmethod
    def _softmax_backward(grad: np.ndarray, s: np.ndarray) -> np.ndarray:
        return s * (grad - (grad * s).sum(axis=-1, keepdims=True))

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

        attn = self._softmax(scores)

        self._Q, self._K, self._V = Q, K, V
        self._attn = attn
        self._scale = scale
        if self.causal:
            self._causal_mask = mask

        return self.W_O.forward(self._merge_heads(attn @ V))

    def backward(self, grad: np.ndarray) -> np.ndarray:
        d_out = self._split_heads(self.W_O.backward(grad))

        d_V = self._attn.transpose(0, 1, 3, 2) @ d_out
        d_attn = d_out @ self._V.transpose(0, 1, 3, 2)

        d_scores = self._softmax_backward(d_attn, self._attn)
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
            self.W_Q.parameters()
            + self.W_K.parameters()
            + self.W_V.parameters()
            + self.W_O.parameters()
        )


# ---------------------------------------------------------------------------
# Feed-Forward sub-layer
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Transformer Block (Pre-LN, GPT-2 style)
# ---------------------------------------------------------------------------

class TransformerBlock(Layer):
    """
    Pre-norm residual block:
      x = x + Attention(LayerNorm1(x))
      x = x + FFN(LayerNorm2(x))
    """

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
        # Correct order: norm wraps the sublayer's backward, not the other way around.
        # For y = x + f(norm(x)):  dx = dy + norm.backward(f.backward(dy))
        grad = grad + self.norm2.backward(self.ffn.backward(grad))
        grad = grad + self.norm1.backward(self.attn.backward(grad))
        return grad

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
        self.pos_emb = SinusoidalPositionalEmbedding(d_model, max_seq_len)
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff, causal=True)
            for _ in range(n_layers)
        ]
        self.norm = LayerNorm(d_model)
        self.head = Linear(d_model, vocab_size)

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        x = self.pos_emb.forward(self.token_emb.forward(tokens))
        for block in self.blocks:
            x = block.forward(x)
        x = self.norm.forward(x)
        return self.head.forward(x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad = self.head.backward(grad)
        grad = self.norm.backward(grad)
        for block in reversed(self.blocks):
            grad = block.backward(grad)
        grad = self.pos_emb.backward(grad)
        self.token_emb.backward(grad)
        return None

    def parameters(self) -> list:
        params = self.token_emb.parameters()
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.norm.parameters())
        params.extend(self.head.parameters())
        return params
