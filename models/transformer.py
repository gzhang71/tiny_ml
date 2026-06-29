import numpy as np
from core.module import Model
from layers.linear import Linear
from layers.normalization import LayerNorm
from layers.embedding import Embedding, SinusoidalPositionalEmbedding
from layers.attention import TransformerBlock


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

    def backward(self, grad: np.ndarray) -> None:
        grad = self.head.backward(grad)
        grad = self.norm.backward(grad)
        for block in reversed(self.blocks):
            grad = block.backward(grad)
        grad = self.pos_emb.backward(grad)
        self.token_emb.backward(grad)

    def parameters(self) -> list:
        params = self.token_emb.parameters()
        for block in self.blocks:
            params.extend(block.parameters())
        params.extend(self.norm.parameters())
        params.extend(self.head.parameters())
        return params
