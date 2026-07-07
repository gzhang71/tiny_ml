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
from core.module import Model
from layers.activations import GeLU
from layers.normalization import LayerNorm
from layers.embedding import Embedding, LearnedPositionalEmbedding
from layers.attention import TransformerBlock, _TiedProjection


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
        self.head = _TiedProjection(self.token_emb)
        self._cache_len = 0  # tokens already in the KV cache

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

    def forward(self, tokens: np.ndarray, use_cache: bool = False) -> np.ndarray:
        offset = self._cache_len if use_cache else 0
        x = self.pos_emb.forward(self.token_emb.forward(tokens), offset=offset)
        for block in self.blocks:
            x = block.forward(x, use_cache=use_cache)
        if use_cache:
            self._cache_len += tokens.shape[1]
        x = self.norm.forward(x)
        return self.head.forward(x)

    def reset_cache(self) -> None:
        for block in self.blocks:
            block.reset_cache()
        self._cache_len = 0

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
        # self.head shares token_emb.W — parameters() returns []
        return params

    def generate(
        self,
        prompt: np.ndarray,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> np.ndarray:
        """Autoregressive token generation with a KV cache.

        Prefill: one parallel pass over the prompt fills the cache.
        Decode: each subsequent pass feeds only the newest token.

        prompt: int array of shape (1, T) or (T,)
        Returns the full sequence including the prompt.
        """
        tokens = np.atleast_2d(prompt)
        self.reset_cache()
        logits = self.forward(tokens, use_cache=True)  # prefill

        for _ in range(max_new_tokens):
            next_logits = logits[0, -1] / temperature

            if top_k is not None:
                threshold = np.sort(next_logits)[-top_k]
                next_logits = np.where(next_logits >= threshold, next_logits, -1e9)

            probs = np.exp(next_logits - next_logits.max())
            probs /= probs.sum()
            next_token = np.random.choice(len(probs), p=probs)
            tokens = np.concatenate([tokens, [[next_token]]], axis=1)
            logits = self.forward(np.array([[next_token]]), use_cache=True)  # decode step

        self.reset_cache()
        return tokens[0]
