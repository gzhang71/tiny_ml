"""
GPT-2 (decoder-only Transformer).

Key differences from the generic Transformer:
  - Learned positional embeddings (not sinusoidal)
  - GeLU activation in feed-forward layers
  - Output projection tied to token embedding (weight tying)
  - Preset configs matching the original paper

Usage:
    model = GPT2.small()
    logits = model.forward(tokens)          # (B, T, vocab_size)
    model.backward(grad)
    token_ids = model.generate(prompt, max_new_tokens=50)
"""
import numpy as np
from tiny_ml.core.module import Layer, Model
from tiny_ml.layers.linear import Linear
from tiny_ml.layers.activations import GeLU
from tiny_ml.layers.normalization import LayerNorm
from tiny_ml.layers.embedding import Embedding, LearnedPositionalEmbedding
from tiny_ml.models.transformer import TransformerBlock


class _TiedProjection(Layer):
    """Output head that reuses the token embedding matrix (W_emb^T).

    Gradient accumulates into the shared embedding Parameter so it is
    updated by a single optimizer entry.
    """

    def __init__(self, embedding: Embedding):
        self._W = embedding.W  # (vocab_size, d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self._W.data.T  # (B, T, vocab_size)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        x_2d = self._x.reshape(-1, self._x.shape[-1])
        grad_2d = grad.reshape(-1, grad.shape[-1])
        self._W.grad += grad_2d.T @ x_2d          # (vocab_size, d_model)
        return (grad_2d @ self._W.data).reshape(self._x.shape)

    def parameters(self) -> list:
        return []  # weight is owned by the Embedding; don't double-count


class GPT2(Model):
    """
    GPT-2 with weight-tied embeddings and GeLU feed-forward layers.

    forward(tokens) → logits (B, T, vocab_size)
    tokens: int array (B, T)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_seq_len: int = 1024,
        d_ff: int | None = None,
    ):
        self.token_emb = Embedding(vocab_size, d_model)
        self.pos_emb = LearnedPositionalEmbedding(max_seq_len, d_model)
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff, causal=True, activation_cls=GeLU)
            for _ in range(n_layers)
        ]
        self.norm = LayerNorm(d_model)
        self.head = _TiedProjection(self.token_emb)  # shares token_emb.W

    # ------------------------------------------------------------------
    # Named configs matching the GPT-2 paper
    # ------------------------------------------------------------------

    @classmethod
    def small(cls) -> "GPT2":
        """117 M parameter GPT-2 (small)."""
        return cls(vocab_size=50257, d_model=768, n_heads=12, n_layers=12)

    @classmethod
    def medium(cls) -> "GPT2":
        """345 M parameter GPT-2 (medium)."""
        return cls(vocab_size=50257, d_model=1024, n_heads=16, n_layers=24)

    @classmethod
    def large(cls) -> "GPT2":
        """762 M parameter GPT-2 (large)."""
        return cls(vocab_size=50257, d_model=1280, n_heads=20, n_layers=36)

    @classmethod
    def xl(cls) -> "GPT2":
        """1.5 B parameter GPT-2 (XL)."""
        return cls(vocab_size=50257, d_model=1600, n_heads=25, n_layers=48)

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        x = self.pos_emb.forward(self.token_emb.forward(tokens))
        for block in self.blocks:
            x = block.forward(x)
        x = self.norm.forward(x)
        return self.head.forward(x)

    def backward(self, grad: np.ndarray) -> None:
        grad = self.head.backward(grad)
        grad = self.norm.backward(grad)
        for block in reversed(self.blocks):
            grad = block.backward(grad)
        grad = self.pos_emb.backward(grad)
        self.token_emb.backward(grad)

    def parameters(self) -> list:
        params = self.token_emb.parameters() + self.pos_emb.parameters()
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.norm.parameters())
        # self.head shares token_emb.W — parameters() returns [] for it
        return params

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: np.ndarray,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> np.ndarray:
        """Autoregressive token generation.

        prompt: int array of shape (1, T) or (T,)
        Returns the full sequence including the prompt.
        """
        tokens = np.atleast_2d(prompt)  # (1, T)
        for _ in range(max_new_tokens):
            logits = self.forward(tokens)          # (1, T, vocab_size)
            next_logits = logits[0, -1] / temperature  # (vocab_size,)

            if top_k is not None:
                threshold = np.sort(next_logits)[-top_k]
                next_logits = np.where(next_logits >= threshold, next_logits, -1e9)

            probs = np.exp(next_logits - next_logits.max())
            probs /= probs.sum()
            next_token = np.random.choice(len(probs), p=probs)
            tokens = np.concatenate([tokens, [[next_token]]], axis=1)

        return tokens[0]
