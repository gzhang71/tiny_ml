from core.backend import BACKEND, xp as np, scatter_add, write_slice
from core.module import Layer
from core.parameter import Parameter
from layers.linear import Linear
from layers.normalization import LayerNorm
from layers.feedforward import FeedForward
from layers.embedding import Embedding, RotaryPositionalEmbedding


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _softmax_backward(grad: np.ndarray, s: np.ndarray) -> np.ndarray:
    return s * (grad - (grad * s).sum(axis=-1, keepdims=True))


class Attention(Layer):
    """Base class for all multi-head attention variants.

    Owns the Q/K/V/O projections, head split/merge, the static KV cache, and
    the standard dense softmax(QKᵀ/√d_k)V computation. Subclasses derive new
    variants by overriding hooks, not `forward`/`backward`:

    - `_attend(Q, K, V) -> O` and `_attend_backward(d_out) -> (dQ, dK, dV)` —
      the uncached attention core between the projections, in per-head layout
      (B, H, T, d_k). The default is the dense masked softmax; FlashAttention
      swaps in the tiled online-softmax kernels.
    - `_score_bias(seq_q, seq_k, q_offset) -> bias | None` and
      `_score_bias_backward(d_scores)` — an additive (n_heads, seq_q, seq_k)
      score bias. The default is none; T5SelfAttention returns its relative
      position bias. Applied in both the uncached and cached paths.
    - `_position_encode(Q, K, offset) -> (Q, K)` and
      `_position_encode_backward(dQ, dK) -> (dQ, dK)` — a per-head transform
      of queries/keys right after the projections, before the attention core
      or the KV cache sees them. The default is identity; RoPEAttention
      rotates by absolute position (`offset` is the cache length when
      decoding, so cached keys are stored already rotated).

    KV cache (`forward(x, use_cache=True)`) is inference-only: new keys/values
    are written into a cache *preallocated* at `max_cache_len` so every decode
    step has identical array shapes (in jax mode a growing cache would
    recompile every op at every step), and queries are masked by absolute
    position, which hides both future tokens and unwritten padded slots.
    `backward` assumes the last forward was uncached.
    """

    def __init__(self, d_model: int, n_heads: int, causal: bool = True,
                 max_cache_len: int = 512):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.causal = causal
        self.max_cache_len = max_cache_len
        self.W_Q = Linear(d_model, d_model)
        self.W_K = Linear(d_model, d_model)
        self.W_V = Linear(d_model, d_model)
        self.W_O = Linear(d_model, d_model)
        self._cache_k: np.ndarray | None = None
        self._cache_v: np.ndarray | None = None
        self._cache_len = 0

    # ---- head bookkeeping ------------------------------------------------

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        B, T, _ = x.shape
        return x.reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        B, H, T, dk = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * dk)

    # ---- KV cache ---------------------------------------------------------

    def reset_cache(self) -> None:
        self._cache_k = None
        self._cache_v = None
        self._cache_len = 0

    # ---- subclass hooks ----------------------------------------------------

    def _score_bias(self, seq_q: int, seq_k: int, q_offset: int = 0) -> np.ndarray | None:
        """Additive (n_heads, seq_q, seq_k) score bias, or None."""
        return None

    def _score_bias_backward(self, d_scores: np.ndarray) -> None:
        """Receives d_scores (B, H, seq_q, seq_k) *before* the 1/√d_k scaling."""

    def _position_encode(self, Q: np.ndarray, K: np.ndarray, offset: int) -> tuple[np.ndarray, np.ndarray]:
        """Position transform of per-head Q/K after the projections."""
        return Q, K

    def _position_encode_backward(self, d_Q: np.ndarray, d_K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return d_Q, d_K

    def _attend(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Uncached attention core: (B, H, T, d_k) → (B, H, T, d_k).

        Saves whatever `_attend_backward` needs.
        """
        seq_q, seq_k = Q.shape[2], K.shape[2]
        scale = np.sqrt(self.d_k)
        scores = Q @ K.transpose(0, 1, 3, 2) / scale

        bias = self._score_bias(seq_q, seq_k)
        if bias is not None:
            scores = scores + bias[None]

        if self.causal:
            mask = np.triu(np.ones((seq_q, seq_k), dtype=bool), k=1)
            scores = np.where(mask, -1e9, scores)
            self._causal_mask = mask

        attn = _softmax(scores)
        self._Q, self._K, self._V, self._attn, self._scale = Q, K, V, attn, scale
        return attn @ V

    def _attend_backward(self, d_out: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Gradient of `_attend`: d_out (B, H, T, d_k) → (dQ, dK, dV)."""
        d_V = self._attn.transpose(0, 1, 3, 2) @ d_out
        d_attn = d_out @ self._V.transpose(0, 1, 3, 2)

        d_scores = _softmax_backward(d_attn, self._attn)
        if self.causal:
            d_scores = np.where(self._causal_mask, 0.0, d_scores)

        self._score_bias_backward(d_scores)
        d_scores = d_scores / self._scale

        d_Q = d_scores @ self._K
        d_K = d_scores.transpose(0, 1, 3, 2) @ self._Q
        return d_Q, d_K, d_V

    # ---- forward / backward -------------------------------------------------

    def forward(self, x: np.ndarray, use_cache: bool = False) -> np.ndarray:
        B, T, _ = x.shape
        Q = self._split_heads(self.W_Q.forward(x))
        K = self._split_heads(self.W_K.forward(x))
        V = self._split_heads(self.W_V.forward(x))
        Q, K = self._position_encode(Q, K, self._cache_len if use_cache else 0)

        if use_cache:
            return self._forward_cached(Q, K, V, B, T)
        return self.W_O.forward(self._merge_heads(self._attend(Q, K, V)))

    def _forward_cached(self, Q, K_new, V_new, B: int, T: int) -> np.ndarray:
        past = self._cache_len
        assert past + T <= self.max_cache_len, (
            f"KV cache overflow ({past + T} > {self.max_cache_len}); "
            "raise max_cache_len / max_seq_len"
        )
        if self._cache_k is None:
            shape = (B, self.n_heads, self.max_cache_len, self.d_k)
            self._cache_k = np.zeros(shape, dtype=K_new.dtype)
            self._cache_v = np.zeros(shape, dtype=V_new.dtype)
        self._cache_k = write_slice(self._cache_k, K_new, past, axis=2)
        self._cache_v = write_slice(self._cache_v, V_new, past, axis=2)
        self._cache_len = past + T

        if BACKEND == "jax":
            # attend over the full padded cache: constant shapes, no recompile
            K, V = self._cache_k, self._cache_v
        else:
            # numpy has no compile cache to protect; skip the padded slots
            K = self._cache_k[:, :, : past + T]
            V = self._cache_v[:, :, : past + T]

        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
        bias = self._score_bias(T, K.shape[2], q_offset=past)
        if bias is not None:
            scores = scores + bias[None]

        # query i sits at absolute position past + i; mask keys beyond it
        # (this also hides the not-yet-written padded slots)
        q_pos = past + np.arange(T)
        mask = np.arange(K.shape[2])[None, :] > q_pos[:, None]
        scores = np.where(mask, -1e9, scores)
        return self.W_O.forward(self._merge_heads(_softmax(scores) @ V))

    def backward(self, grad: np.ndarray) -> np.ndarray:
        d_out = self._split_heads(self.W_O.backward(grad))
        d_Q, d_K, d_V = self._attend_backward(d_out)
        d_Q, d_K = self._position_encode_backward(d_Q, d_K)
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


class MultiHeadAttention(Attention):
    """Scaled dot-product multi-head self-attention with optional causal mask.

    The base class's default behavior with no extra hooks — kept as its own
    name so models read as "standard attention" and variants read as
    deviations from it.
    """


class RoPEAttention(Attention):
    """Multi-head self-attention with rotary position embedding (RoPE).

    Derives from `Attention` via the position hooks: Q and K are rotated by
    their absolute positions right after the projections (`offset` = cache
    length when decoding, so cached keys are stored already rotated and the
    relative-position property holds across decode steps). The backward hook
    applies the inverse rotation. A model using this needs no additive
    positional embedding at the input.
    """

    def __init__(self, d_model: int, n_heads: int, causal: bool = True,
                 max_cache_len: int = 512, rope_base: float = 10000.0):
        super().__init__(d_model, n_heads, causal=causal, max_cache_len=max_cache_len)
        self.rope = RotaryPositionalEmbedding(self.d_k, max_seq_len=max_cache_len,
                                              base=rope_base)

    def _position_encode(self, Q: np.ndarray, K: np.ndarray, offset: int) -> tuple[np.ndarray, np.ndarray]:
        return self.rope.forward(Q, offset), self.rope.forward(K, offset)

    def _position_encode_backward(self, d_Q: np.ndarray, d_K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.rope.backward(d_Q), self.rope.backward(d_K)


class TransformerBlock(Layer):
    """Pre-norm residual block: x = x + Attn(LN(x));  x = x + FFN(LN(x))."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int | None = None,
        causal: bool = True,
        activation_cls=None,
        max_cache_len: int = 512,
        attention_cls: type[Attention] = MultiHeadAttention,
        ffn_cls: type = FeedForward,
    ):
        self.norm1 = LayerNorm(d_model)
        self.attn = attention_cls(d_model, n_heads, causal=causal,
                                  max_cache_len=max_cache_len)
        self.norm2 = LayerNorm(d_model)
        self.ffn = ffn_cls(d_model, d_ff, activation_cls=activation_cls)

    def forward(self, x: np.ndarray, use_cache: bool = False) -> np.ndarray:
        x = x + self.attn.forward(self.norm1.forward(x), use_cache=use_cache)
        x = x + self.ffn.forward(self.norm2.forward(x))
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad = grad + self.norm2.backward(self.ffn.backward(grad))
        grad = grad + self.norm1.backward(self.attn.backward(grad))
        return grad

    def reset_cache(self) -> None:
        self.attn.reset_cache()

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

    def forward(self, seq_q: int, seq_k: int, q_offset: int = 0) -> np.ndarray:
        q_pos = np.arange(q_offset, q_offset + seq_q)[:, None]
        k_pos = np.arange(seq_k)[None, :]
        self._buckets = self._bucket(q_pos - k_pos)
        return self.W.data[:, self._buckets]

    def backward(self, grad: np.ndarray) -> None:
        for h in range(self.n_heads):
            self.W.grad = scatter_add(self.W.grad, (h, self._buckets), grad[h])


class T5SelfAttention(Attention):
    """Multi-head self-attention with relative position bias (T5 style).

    Derives from `Attention` via the score-bias hooks: the relative position
    bias is added to the scores in both the uncached and cached paths, and
    its gradient is the un-scaled d_scores summed over the batch.
    """

    def __init__(self, d_model: int, n_heads: int, causal: bool, n_buckets: int = 32,
                 max_cache_len: int = 512):
        super().__init__(d_model, n_heads, causal=causal, max_cache_len=max_cache_len)
        self.rel_bias = RelativePositionBias(n_heads, n_buckets, bidirectional=not causal)

    def _score_bias(self, seq_q: int, seq_k: int, q_offset: int = 0) -> np.ndarray:
        return self.rel_bias.forward(seq_q, seq_k, q_offset=q_offset)

    def _score_bias_backward(self, d_scores: np.ndarray) -> None:
        self.rel_bias.backward(d_scores.sum(axis=0))

    def parameters(self) -> list:
        return super().parameters() + self.rel_bias.parameters()


class CrossAttention(Attention):
    """Multi-head cross-attention: Q ← decoder, K/V ← encoder.

    Reuses the base class's projections and attention core (uncausal, no
    mask), but takes two inputs, so `forward`/`backward` are overridden and
    `backward` returns a (d_x_dec, d_x_enc) tuple. The cache holds the
    encoder K/V, computed on the first `use_cache=True` call and reused on
    every subsequent decode step (the encoder output never changes during
    generation). Inference-only, like the self-attention KV cache.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__(d_model, n_heads, causal=False)

    def forward(self, x_dec: np.ndarray, x_enc: np.ndarray, use_cache: bool = False) -> np.ndarray:
        Q = self._split_heads(self.W_Q.forward(x_dec))
        if use_cache and self._cache_k is not None:
            K, V = self._cache_k, self._cache_v
        else:
            K = self._split_heads(self.W_K.forward(x_enc))
            V = self._split_heads(self.W_V.forward(x_enc))
            if use_cache:
                self._cache_k, self._cache_v = K, V

        return self.W_O.forward(self._merge_heads(self._attend(Q, K, V)))

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (d_x_dec, d_x_enc)."""
        d_out = self._split_heads(self.W_O.backward(grad))
        d_Q, d_K, d_V = self._attend_backward(d_out)
        return (
            self.W_Q.backward(self._merge_heads(d_Q)),
            self.W_K.backward(self._merge_heads(d_K)) + self.W_V.backward(self._merge_heads(d_V)),
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
