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
from core.module import Layer, Model
from layers.normalization import LayerNorm
from layers.embedding import Embedding
from layers.feedforward import FeedForward
from layers.attention import T5SelfAttention, CrossAttention, RelativePositionBias, _TiedProjection


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

    def forward(self, x: np.ndarray, enc_out: np.ndarray, use_cache: bool = False) -> np.ndarray:
        x = x + self.self_attn.forward(self.norm1.forward(x), use_cache=use_cache)
        x = x + self.cross_attn.forward(self.norm2.forward(x), enc_out, use_cache=use_cache)
        x = x + self.ffn.forward(self.norm3.forward(x))
        return x

    def reset_cache(self) -> None:
        self.self_attn.reset_cache()
        self.cross_attn.reset_cache()

    def backward(self, grad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (d_x, d_enc_out)."""
        grad = grad + self.norm3.backward(self.ffn.backward(grad))
        d_cross_dec, d_enc_out = self.cross_attn.backward(grad)
        grad = grad + self.norm2.backward(d_cross_dec)
        grad = grad + self.norm1.backward(self.self_attn.backward(grad))
        return grad, d_enc_out

    def parameters(self) -> list:
        return (
            self.norm1.parameters() + self.self_attn.parameters()
            + self.norm2.parameters() + self.cross_attn.parameters()
            + self.norm3.parameters() + self.ffn.parameters()
        )


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
        self.head = _TiedProjection(self.shared_emb)

        self._src_tokens: np.ndarray | None = None
        self._tgt_tokens: np.ndarray | None = None
        self._enc_out: np.ndarray | None = None

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

    def encode(self, src_tokens: np.ndarray) -> np.ndarray:
        x = self.shared_emb.forward(src_tokens)
        for block in self.encoder_blocks:
            x = block.forward(x)
        return self.encoder_norm.forward(x)

    def decode(self, tgt_tokens: np.ndarray, enc_out: np.ndarray, use_cache: bool = False) -> np.ndarray:
        x = self.shared_emb.forward(tgt_tokens)
        for block in self.decoder_blocks:
            x = block.forward(x, enc_out, use_cache=use_cache)
        x = self.decoder_norm.forward(x)
        return self.head.forward(x)

    def reset_cache(self) -> None:
        for block in self.decoder_blocks:
            block.reset_cache()

    def forward(self, src_tokens: np.ndarray, tgt_tokens: np.ndarray) -> np.ndarray:
        self._src_tokens = src_tokens
        self._tgt_tokens = tgt_tokens
        self._enc_out = self.encode(src_tokens)
        return self.decode(tgt_tokens, self._enc_out)

    def backward(self, grad: np.ndarray) -> None:
        grad = self.head.backward(grad)
        grad = self.decoder_norm.backward(grad)
        d_enc_out_total = np.zeros_like(self._enc_out)
        for block in reversed(self.decoder_blocks):
            grad, d_enc_out = block.backward(grad)
            d_enc_out_total += d_enc_out
        # shared_emb._tokens was overwritten by tgt forward; bypass cached method
        np.add.at(self.shared_emb.W.grad, self._tgt_tokens, grad)

        grad_enc = self.encoder_norm.backward(d_enc_out_total)
        for block in reversed(self.encoder_blocks):
            grad_enc = block.backward(grad_enc)
        np.add.at(self.shared_emb.W.grad, self._src_tokens, grad_enc)

    def generate(
        self,
        src_tokens: np.ndarray,
        max_new_tokens: int = 50,
        start_token: int = 0,
        eos_token: int | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> np.ndarray:
        """Autoregressive decoding with KV caches.

        The encoder runs once; its K/V are cached inside each cross-attention.
        Decoder self-attention caches grow one token per step, so each step
        feeds only the newest token.

        src_tokens: int array of shape (1, T_src) or (T_src,)
        Returns the generated target sequence (start token excluded).
        """
        src = np.atleast_2d(src_tokens)
        self.reset_cache()
        enc_out = self.encode(src)

        token = np.array([[start_token]])
        generated: list[int] = []
        for _ in range(max_new_tokens):
            logits = self.decode(token, enc_out, use_cache=True)
            next_logits = logits[0, -1] / temperature

            if top_k is not None:
                threshold = np.sort(next_logits)[-top_k]
                next_logits = np.where(next_logits >= threshold, next_logits, -1e9)

            probs = np.exp(next_logits - next_logits.max())
            probs /= probs.sum()
            next_token = int(np.random.choice(len(probs), p=probs))
            if eos_token is not None and next_token == eos_token:
                break
            generated.append(next_token)
            token = np.array([[next_token]])

        self.reset_cache()
        return np.array(generated)

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
