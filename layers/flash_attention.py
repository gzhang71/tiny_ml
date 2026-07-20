"""FlashAttention (v1) and FlashAttention-2: tiled attention with online softmax.

Both compute *exactly* the same function as `MultiHeadAttention` — softmax
(QKᵀ/√d)V — but never materialize the (T, T) attention matrix. Scores are
computed one (block_q × block_k) tile at a time and folded into a running
softmax (running row-max `m`, running denominator `l`), so the extra memory
per head is O(T) instead of O(T²). The backward pass recomputes each tile's
probabilities from the saved logsumexp `L = m + log l` instead of storing them
(the paper's recomputation trick), using the identity

    rowsum(P ∘ dP) = rowsum(dO ∘ O)

so dS for a tile needs only that tile plus two O(T) vectors.

The two versions differ only in loop order and how often the accumulator is
rescaled:

- **v1** (`FlashAttention`): outer loop over K/V blocks, inner over Q blocks.
  Every visit renormalizes the output block by the updated denominator
  (`O ← (α·l·O + β·P·V) / l_new`) — one extra multiply *and* divide per tile.
- **v2** (`FlashAttention2`): outer loop over Q blocks, inner over K/V blocks.
  The accumulator stays *unnormalized*; each step only applies the max-shift
  correction `exp(m_old − m_new)`, and the single division by `l` happens
  once per Q block after its inner loop finishes.

In this library the tiles are plain numpy ops, so there is no speed win —
the point is the algorithm (and the O(T) memory footprint is real). Both are
drop-in replacements for `MultiHeadAttention`; `use_cache=True` decoding
falls back to the standard cached path, where tiling buys nothing.
"""
from core.backend import xp as np
from layers.attention import Attention


def _masked_scores(Q_blk, K_blk, scale, causal, i0, i1, j0, j1, key_pad=None):
    """Scaled scores for one tile, with causal and padding masks applied in-tile.

    Masked entries get -1e9 (matching the rest of the library), which
    underflows to an exact 0 after the exp — including in the backward
    recomputation, so dS needs no separate mask.

    `key_pad` is the layer's (B, 1, 1, T_k) padding mask; only this tile's
    key slice is applied, and it broadcasts over heads and queries.
    """
    S = Q_blk @ K_blk.transpose(0, 1, 3, 2) / scale
    if causal and j1 - 1 > i0:
        mask = np.arange(j0, j1)[None, :] > np.arange(i0, i1)[:, None]
        S = np.where(mask, -1e9, S)
    if key_pad is not None:
        S = np.where(key_pad[..., j0:j1], -1e9, S)
    return S


def _flash_attention_forward(Q, K, V, scale, causal, block_q, block_k, key_pad=None):
    """FlashAttention v1 forward. Returns (O, L) with L the per-row logsumexp.

    Outer loop over K/V blocks: each Q block's output is renormalized by the
    updated softmax denominator on every visit (`/ l_new`), which is the per-
    tile division v2 gets rid of.
    """
    B, H, T, d = Q.shape
    Tk = K.shape[2]
    q_starts = list(range(0, T, block_q))
    O, m, l = [], [], []
    for i0 in q_starts:
        br = min(block_q, T - i0)
        O.append(np.zeros((B, H, br, d), dtype=Q.dtype))
        m.append(np.full((B, H, br), -np.inf, dtype=Q.dtype))
        l.append(np.zeros((B, H, br), dtype=Q.dtype))

    for j0 in range(0, Tk, block_k):
        j1 = min(j0 + block_k, Tk)
        Kj, Vj = K[:, :, j0:j1], V[:, :, j0:j1]
        for bi, i0 in enumerate(q_starts):
            i1 = min(i0 + block_q, T)
            if causal and j0 > i1 - 1:
                continue  # tile entirely above the diagonal
            S = _masked_scores(Q[:, :, i0:i1], Kj, scale, causal, i0, i1, j0, j1, key_pad)
            m_blk = S.max(axis=-1)
            P = np.exp(S - m_blk[..., None])
            l_blk = P.sum(axis=-1)
            m_new = np.maximum(m[bi], m_blk)
            alpha = np.exp(m[bi] - m_new)
            beta = np.exp(m_blk - m_new)
            l_new = alpha * l[bi] + beta * l_blk
            O[bi] = (
                (alpha * l[bi])[..., None] * O[bi] + beta[..., None] * (P @ Vj)
            ) / l_new[..., None]
            m[bi], l[bi] = m_new, l_new

    return np.concatenate(O, axis=2), np.concatenate(m, axis=2) + np.log(
        np.concatenate(l, axis=2)
    )


def _flash_attention2_forward(Q, K, V, scale, causal, block_q, block_k, key_pad=None):
    """FlashAttention-2 forward. Returns (O, L) with L the per-row logsumexp.

    Outer loop over Q blocks: the accumulator stays unnormalized inside the
    inner loop (only max-shift corrections), and the division by the softmax
    denominator happens once per Q block at the end.
    """
    B, H, T, d = Q.shape
    Tk = K.shape[2]
    outs, lses = [], []
    for i0 in range(0, T, block_q):
        i1 = min(i0 + block_q, T)
        Qi = Q[:, :, i0:i1]
        m = np.full((B, H, i1 - i0), -np.inf, dtype=Q.dtype)
        l = np.zeros((B, H, i1 - i0), dtype=Q.dtype)
        acc = np.zeros((B, H, i1 - i0, d), dtype=Q.dtype)
        for j0 in range(0, Tk, block_k):
            if causal and j0 > i1 - 1:
                break  # this and all later tiles are above the diagonal
            j1 = min(j0 + block_k, Tk)
            S = _masked_scores(Qi, K[:, :, j0:j1], scale, causal, i0, i1, j0, j1, key_pad)
            m_new = np.maximum(m, S.max(axis=-1))
            P = np.exp(S - m_new[..., None])
            corr = np.exp(m - m_new)
            l = corr * l + P.sum(axis=-1)
            acc = corr[..., None] * acc + P @ V[:, :, j0:j1]
            m = m_new
        outs.append(acc / l[..., None])
        lses.append(m + np.log(l))
    return np.concatenate(outs, axis=2), np.concatenate(lses, axis=2)


def _flash_attention_backward(dO, Q, K, V, O, L, scale, causal, block_q, block_k,
                              key_pad=None):
    """Tiled backward shared by v1 and v2. Returns (dQ, dK, dV).

    Recomputes each tile's probabilities as P = exp(S − L) from the saved
    logsumexp instead of storing the attention matrix, and replaces the
    softmax-jacobian row sum with D = rowsum(dO ∘ O).
    """
    B, H, T, d = Q.shape
    Tk = K.shape[2]
    D = (dO * O).sum(axis=-1)
    q_starts = list(range(0, T, block_q))
    dQ = [np.zeros((B, H, min(block_q, T - i0), d), dtype=Q.dtype) for i0 in q_starts]
    dK_parts, dV_parts = [], []

    for j0 in range(0, Tk, block_k):
        j1 = min(j0 + block_k, Tk)
        Kj, Vj = K[:, :, j0:j1], V[:, :, j0:j1]
        dKj = np.zeros((B, H, j1 - j0, d), dtype=Q.dtype)
        dVj = np.zeros((B, H, j1 - j0, d), dtype=Q.dtype)
        for bi, i0 in enumerate(q_starts):
            i1 = min(i0 + block_q, T)
            if causal and j0 > i1 - 1:
                continue
            Qi, dOi = Q[:, :, i0:i1], dO[:, :, i0:i1]
            S = _masked_scores(Qi, Kj, scale, causal, i0, i1, j0, j1, key_pad)
            P = np.exp(S - L[:, :, i0:i1][..., None])
            dVj = dVj + P.transpose(0, 1, 3, 2) @ dOi
            dP = dOi @ Vj.transpose(0, 1, 3, 2)
            dS = P * (dP - D[:, :, i0:i1][..., None]) / scale
            dQ[bi] = dQ[bi] + dS @ Kj
            dKj = dKj + dS.transpose(0, 1, 3, 2) @ Qi
        dK_parts.append(dKj)
        dV_parts.append(dVj)

    return (
        np.concatenate(dQ, axis=2),
        np.concatenate(dK_parts, axis=2),
        np.concatenate(dV_parts, axis=2),
    )


class FlashAttention(Attention):
    """Multi-head attention computed with the FlashAttention (v1) tiling.

    Numerically identical to `MultiHeadAttention` (same projections, mask,
    and gradients) but the (T, T) attention matrix is never formed: forward
    saves only the output O and the logsumexp L, and backward recomputes
    probabilities tile by tile. Derives from `Attention` by overriding just
    the `_attend` / `_attend_backward` core; `use_cache=True` decoding uses
    the inherited standard cached path, where tiling buys nothing.
    """

    def __init__(self, d_model: int, n_heads: int, causal: bool = True,
                 max_cache_len: int = 512, n_kv_heads: int | None = None,
                 block_q: int = 64, block_k: int = 64):
        super().__init__(d_model, n_heads, causal=causal, max_cache_len=max_cache_len,
                         n_kv_heads=n_kv_heads)
        self.block_q = block_q
        self.block_k = block_k

    _flash_forward = staticmethod(_flash_attention_forward)

    def _attend(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        scale = np.sqrt(self.d_k)
        O, L = self._flash_forward(Q, K, V, scale, self.causal,
                                   self.block_q, self.block_k, self._key_pad)
        self._Q, self._K, self._V, self._O, self._L, self._scale = Q, K, V, O, L, scale
        return O

    def _attend_backward(self, d_out: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _flash_attention_backward(
            d_out, self._Q, self._K, self._V, self._O, self._L,
            self._scale, self.causal, self.block_q, self.block_k, self._key_pad,
        )


class FlashAttention2(FlashAttention):
    """Multi-head attention computed with the FlashAttention-2 tiling.

    Same contract as `FlashAttention`; only the forward loop structure
    differs (Q-block outer loop, unnormalized accumulator, one division per
    Q block). The backward is shared.
    """

    _flash_forward = staticmethod(_flash_attention2_forward)
