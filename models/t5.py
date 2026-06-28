"""
T5 encoder-decoder Transformer.

Key differences from GPT-2:
  - Encoder + decoder (separate stacks)
  - Bidirectional encoder (no causal mask)
  - Decoder has causal self-attention PLUS cross-attention to encoder output
  - Relative position bias instead of absolute positional embeddings
  - Shared token embedding between encoder and decoder, tied to output head
  - ReLU feed-forward (original T5; swap to GeLU for T5v1.1)

Usage:
    model = T5.small()
    logits = model.forward(src_tokens, tgt_tokens)   # (B, T_dec, vocab_size)
    model.backward(grad)
"""
import numpy as np
from tiny_ml.core.module import Layer, Model
from tiny_ml.core.prameter import Parameter
from tiny_ml.layers.linear import Linear
from tiny_ml.layers.normalization import LayerNorm
from tiny_ml.layers.embedding import Embedding
from tiny_ml.models.transformer import FeedForward


# ---------------------------------------------------------------------------
# Relative position bias (T5 style)
# ---------------------------------------------------------------------------

class RelativePositionBias(Layer):
    """Learnable scalar bias per (head, relative-position-bucket) pair.

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
        self._buckets = self._bucket(q_pos - k_pos)         # (seq_q, seq_k)
        return self.W.data[:, self._buckets]                 # (n_heads, seq_q, seq_k)

    def backward(self, grad: np.ndarray) -> None:
        """grad: (n_heads, seq_q, seq_k) — summed over batch by caller."""
        for h in range(self.n_heads):
            np.add.at(self.W.grad[h], self._buckets, grad[h])


# ---------------------------------------------------------------------------
# T5 Self-Attention (with relative position bias)
# ---------------------------------------------------------------------------

class T5SelfAttention(Layer):
    """Multi-head self-attention + relative position bias."""

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

    def _split(self, x):
        B, T, _ = x.shape
        return x.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

    def _merge(self, x):
        B, H, T, dk = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * dk)

    @staticmethod
    def _softmax(x):
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)

    @staticmethod
    def _softmax_bwd(grad, s):
        return s * (grad - (grad * s).sum(axis=-1, keepdims=True))

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, T, _ = x.shape
        Q = self._split(self.W_Q.forward(x))
        K = self._split(self.W_K.forward(x))
        V = self._split(self.W_V.forward(x))

        scale = np.sqrt(self.d_k)
        scores = Q @ K.transpose(0, 1, 3, 2) / scale        # (B, H, T, T)
        scores = scores + self.rel_bias.forward(T, T)[None]  # broadcast over batch

        if self.causal:
            mask = np.triu(np.ones((T, T), dtype=bool), k=1)
            scores = np.where(mask, -1e9, scores)
            self._causal_mask = mask

        attn = self._softmax(scores)
        self._Q, self._K, self._V, self._attn, self._scale = Q, K, V, attn, scale
        return self.W_O.forward(self._merge(attn @ V))

    def backward(self, grad: np.ndarray) -> np.ndarray:
        d_out = self._split(self.W_O.backward(grad))

        d_V = self._attn.transpose(0, 1, 3, 2) @ d_out
        d_attn = d_out @ self._V.transpose(0, 1, 3, 2)

        d_scores = self._softmax_bwd(d_attn, self._attn)
        if self.causal:
            d_scores = np.where(self._causal_mask, 0.0, d_scores)

        self.rel_bias.backward(d_scores.sum(axis=0))  # (H, T, T)
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


# ---------------------------------------------------------------------------
# Cross-Attention (Q from decoder, K/V from encoder)
# ---------------------------------------------------------------------------

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

    def _split(self, x):
        B, T, _ = x.shape
        return x.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

    def _merge(self, x):
        B, H, T, dk = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * dk)

    @staticmethod
    def _softmax(x):
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)

    @staticmethod
    def _softmax_bwd(grad, s):
        return s * (grad - (grad * s).sum(axis=-1, keepdims=True))

    def forward(self, x_dec: np.ndarray, x_enc: np.ndarray) -> np.ndarray:
        scale = np.sqrt(self.d_k)
        Q = self._split(self.W_Q.forward(x_dec))
        K = self._split(self.W_K.forward(x_enc))
        V = self._split(self.W_V.forward(x_enc))

        attn = self._softmax(Q @ K.transpose(0, 1, 3, 2) / scale)
        self._Q, self._K, self._V, self._attn, self._scale = Q, K, V, attn, scale
        return self.W_O.forward(self._merge(attn @ V))

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (d_x_dec, d_x_enc)."""
        d_out = self._split(self.W_O.backward(grad))

        d_V = self._attn.transpose(0, 1, 3, 2) @ d_out
        d_attn = d_out @ self._V.transpose(0, 1, 3, 2)
        d_scores = self._softmax_bwd(d_attn, self._attn) / self._scale

        d_Q = d_scores @ self._K
        d_K = d_scores.transpose(0, 1, 3, 2) @ self._Q

        d_x_dec = self.W_Q.backward(self._merge(d_Q))
        d_x_enc = (
            self.W_K.backward(self._merge(d_K))
            + self.W_V.backward(self._merge(d_V))
        )
        return d_x_dec, d_x_enc

    def parameters(self) -> list:
        return (
            self.W_Q.parameters() + self.W_K.parameters()
            + self.W_V.parameters() + self.W_O.parameters()
        )


# ---------------------------------------------------------------------------
# T5 Tied output projection
# ---------------------------------------------------------------------------

class _TiedProjection(Layer):
    """Output head sharing the embedding matrix (W_emb @ x^T)."""

    def __init__(self, embedding: Embedding):
        self._W = embedding.W  # (vocab_size, d_model)

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


# ---------------------------------------------------------------------------
# T5 Encoder / Decoder blocks
# ---------------------------------------------------------------------------

class T5EncoderBlock(Layer):
    """Pre-norm: x = x + SelfAttn(LN(x));  x = x + FFN(LN(x))"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_buckets: int = 32):
        self.norm1 = LayerNorm(d_model)
        self.self_attn = T5SelfAttention(d_model, n_heads, causal=False, n_buckets=n_buckets)
        self.norm2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x + self.self_attn.forward(self.norm1.forward(x))
        x = x + self.ffn.forward(self.norm2.forward(x))
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad = grad + self.norm2.backward(self.ffn.backward(grad))
        grad = grad + self.norm1.backward(self.self_attn.backward(grad))
        return grad

    def parameters(self) -> list:
        return (
            self.norm1.parameters() + self.self_attn.parameters()
            + self.norm2.parameters() + self.ffn.parameters()
        )


class T5DecoderBlock(Layer):
    """Pre-norm with three sub-layers: causal self-attn, cross-attn, FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_buckets: int = 32):
        self.norm1 = LayerNorm(d_model)
        self.self_attn = T5SelfAttention(d_model, n_heads, causal=True, n_buckets=n_buckets)
        self.norm2 = LayerNorm(d_model)
        self.cross_attn = CrossAttention(d_model, n_heads)
        self.norm3 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: np.ndarray, enc_out: np.ndarray) -> np.ndarray:
        x = x + self.self_attn.forward(self.norm1.forward(x))
        x = x + self.cross_attn.forward(self.norm2.forward(x), enc_out)
        x = x + self.ffn.forward(self.norm3.forward(x))
        return x

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (d_x, d_enc_out)."""
        # FFN residual
        grad = grad + self.norm3.backward(self.ffn.backward(grad))
        # Cross-attention residual
        d_cross_dec, d_enc_out = self.cross_attn.backward(grad)
        grad = grad + self.norm2.backward(d_cross_dec)
        # Self-attention residual
        grad = grad + self.norm1.backward(self.self_attn.backward(grad))
        return grad, d_enc_out

    def parameters(self) -> list:
        return (
            self.norm1.parameters() + self.self_attn.parameters()
            + self.norm2.parameters() + self.cross_attn.parameters()
            + self.norm3.parameters() + self.ffn.parameters()
        )


# ---------------------------------------------------------------------------
# T5 model
# ---------------------------------------------------------------------------

class T5(Model):
    """
    T5 encoder-decoder transformer with shared token embeddings.

    forward(src_tokens, tgt_tokens) → logits (B, T_dec, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        d_ff: int | None = None,
        n_buckets: int = 32,
    ):
        d_ff = d_ff or 4 * d_model
        self.shared_emb = Embedding(vocab_size, d_model)
        self.encoder_blocks = [
            T5EncoderBlock(d_model, n_heads, d_ff, n_buckets)
            for _ in range(n_encoder_layers)
        ]
        self.encoder_norm = LayerNorm(d_model)
        self.decoder_blocks = [
            T5DecoderBlock(d_model, n_heads, d_ff, n_buckets)
            for _ in range(n_decoder_layers)
        ]
        self.decoder_norm = LayerNorm(d_model)
        self.head = _TiedProjection(self.shared_emb)  # shares shared_emb.W

        # cached for backward
        self._src_tokens: np.ndarray | None = None
        self._tgt_tokens: np.ndarray | None = None
        self._enc_out: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Named configs
    # ------------------------------------------------------------------

    @classmethod
    def small(cls) -> "T5":
        """T5-small: 60 M params."""
        return cls(vocab_size=32128, d_model=512, n_heads=8,
                   n_encoder_layers=6, n_decoder_layers=6, d_ff=2048)

    @classmethod
    def base(cls) -> "T5":
        """T5-base: 220 M params."""
        return cls(vocab_size=32128, d_model=768, n_heads=12,
                   n_encoder_layers=12, n_decoder_layers=12, d_ff=3072)

    @classmethod
    def large(cls) -> "T5":
        """T5-large: 770 M params."""
        return cls(vocab_size=32128, d_model=1024, n_heads=16,
                   n_encoder_layers=24, n_decoder_layers=24, d_ff=4096)

    # ------------------------------------------------------------------
    # Encode / decode helpers
    # ------------------------------------------------------------------

    def encode(self, src_tokens: np.ndarray) -> np.ndarray:
        x = self.shared_emb.forward(src_tokens)
        for block in self.encoder_blocks:
            x = block.forward(x)
        return self.encoder_norm.forward(x)

    def decode(self, tgt_tokens: np.ndarray, enc_out: np.ndarray) -> np.ndarray:
        x = self.shared_emb.forward(tgt_tokens)
        for block in self.decoder_blocks:
            x = block.forward(x, enc_out)
        x = self.decoder_norm.forward(x)
        return self.head.forward(x)

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def forward(self, src_tokens: np.ndarray, tgt_tokens: np.ndarray) -> np.ndarray:
        self._src_tokens = src_tokens
        self._tgt_tokens = tgt_tokens
        self._enc_out = self.encode(src_tokens)
        return self.decode(tgt_tokens, self._enc_out)

    def backward(self, grad: np.ndarray) -> None:
        # --- decoder backward ---
        grad = self.head.backward(grad)
        grad = self.decoder_norm.backward(grad)
        d_enc_out_total = np.zeros_like(self._enc_out)
        for block in reversed(self.decoder_blocks):
            grad, d_enc_out = block.backward(grad)
            d_enc_out_total += d_enc_out
        # scatter-add decoder embedding grad directly (shared_emb._tokens was
        # overwritten by the tgt forward pass, so we bypass the cached method)
        np.add.at(self.shared_emb.W.grad, self._tgt_tokens, grad)

        # --- encoder backward ---
        grad_enc = self.encoder_norm.backward(d_enc_out_total)
        for block in reversed(self.encoder_blocks):
            grad_enc = block.backward(grad_enc)
        np.add.at(self.shared_emb.W.grad, self._src_tokens, grad_enc)

    def parameters(self) -> list:
        params = self.shared_emb.parameters()
        for block in self.encoder_blocks:
            params.extend(block.parameters())
        params.extend(self.encoder_norm.parameters())
        for block in self.decoder_blocks:
            params.extend(block.parameters())
        params.extend(self.decoder_norm.parameters())
        # self.head shares shared_emb.W — returns []
        return params
