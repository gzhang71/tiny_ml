"""Equivalence tests for the claims the architecture docs make.

Gradient checks prove each backward matches its own forward. These tests prove
the *forwards* that are supposed to agree actually do:

- FlashAttention computes exact attention, so it must equal MultiHeadAttention
- the KV cache is exact, so incremental decoding must equal a full forward
- the jax backend at float64 must match numpy bit-for-near-bit
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as onp

from core.backend import BACKEND, to_numpy
from layers.attention import MultiHeadAttention, TransformerBlock
from layers.flash_attention import FlashAttention, FlashAttention2
from models.gpt2 import GPT2
from models.t5 import T5
from training.checkpoint import get_state, set_state, average_states

REPO_ROOT = Path(__file__).resolve().parent.parent


def _rand(*shape, seed: int = 0) -> onp.ndarray:
    return onp.random.RandomState(seed).randn(*shape)


def _copy_params(src, dst) -> None:
    """Copy parameters positionally (same architecture ⇒ same order)."""
    for p_src, p_dst in zip(src.parameters(), dst.parameters(), strict=True):
        p_dst.data = onp.array(to_numpy(p_src.data))


def _max_abs_diff(a, b) -> float:
    return float(onp.max(onp.abs(onp.asarray(to_numpy(a)) - onp.asarray(to_numpy(b)))))


# ---- FlashAttention is exact ------------------------------------------------

def _check_flash_matches_dense(flash_cls, causal: bool):
    dense = MultiHeadAttention(16, 4, causal=causal)
    flash = flash_cls(16, 4, causal=causal)
    _copy_params(dense, flash)

    # sequence length deliberately not a multiple of the tile size, to exercise the
    # ragged final block of the tiled loop
    x = _rand(2, 7, 16)
    out_dense = dense.forward(x)
    out_flash = flash.forward(x)
    err = _max_abs_diff(out_dense, out_flash)
    assert err < 1e-10, f"{flash_cls.__name__} forward differs from dense by {err:.3e}"

    grad = _rand(2, 7, 16, seed=1)
    dx_dense = dense.backward(grad)
    dx_flash = flash.backward(grad)
    err = _max_abs_diff(dx_dense, dx_flash)
    assert err < 1e-10, f"{flash_cls.__name__} backward differs from dense by {err:.3e}"

    for i, (p_dense, p_flash) in enumerate(
        zip(dense.parameters(), flash.parameters(), strict=True)
    ):
        err = _max_abs_diff(p_dense.grad, p_flash.grad)
        assert err < 1e-10, f"{flash_cls.__name__} param[{i}] grad differs by {err:.3e}"


def test_flash_attention_equals_dense_causal():
    _check_flash_matches_dense(FlashAttention, causal=True)


def test_flash_attention_equals_dense_noncausal():
    _check_flash_matches_dense(FlashAttention, causal=False)


def test_flash_attention_2_equals_dense_causal():
    _check_flash_matches_dense(FlashAttention2, causal=True)


def test_flash_attention_2_equals_dense_noncausal():
    _check_flash_matches_dense(FlashAttention2, causal=False)


# ---- the KV cache is exact ---------------------------------------------------

def test_kv_cache_matches_full_forward():
    """Token-by-token cached decoding must reproduce a single full forward."""
    onp.random.seed(0)
    model = GPT2(vocab_size=32, d_model=16, n_heads=4, n_layers=2, max_seq_len=16)
    tokens = onp.array([[3, 14, 7, 1, 9]])

    full = to_numpy(model.forward(tokens))

    model.reset_cache()
    incremental = []
    for t in range(tokens.shape[1]):
        step = model.forward(tokens[:, t : t + 1], use_cache=True)
        incremental.append(to_numpy(step)[:, -1])
    model.reset_cache()

    err = _max_abs_diff(full[0], onp.stack(incremental, axis=1)[0])
    assert err < 1e-10, f"cached decode differs from full forward by {err:.3e}"


def test_kv_cache_prefill_then_decode():
    """The generate() pattern: prefill a prompt, then decode one at a time."""
    onp.random.seed(0)
    model = GPT2(vocab_size=32, d_model=16, n_heads=4, n_layers=2, max_seq_len=16)
    tokens = onp.array([[3, 14, 7, 1, 9, 2]])

    full = to_numpy(model.forward(tokens))

    model.reset_cache()
    prefill = to_numpy(model.forward(tokens[:, :4], use_cache=True))
    err = _max_abs_diff(full[:, :4], prefill)
    assert err < 1e-10, f"prefill differs by {err:.3e}"

    for t in (4, 5):
        step = to_numpy(model.forward(tokens[:, t : t + 1], use_cache=True))
        err = _max_abs_diff(full[:, t], step[:, 0])
        assert err < 1e-10, f"decode step {t} differs by {err:.3e}"
    model.reset_cache()


def test_kv_cache_rope_matches_full_forward():
    """RoPE stores already-rotated keys in the cache — the risky variant."""
    onp.random.seed(0)
    from layers.attention import RoPEAttention

    block = TransformerBlock(16, 4, d_ff=32, causal=True,
                             attention_cls=RoPEAttention, max_cache_len=16)
    x = _rand(1, 5, 16)
    full = to_numpy(block.forward(x))

    block.reset_cache()
    steps = [to_numpy(block.forward(x[:, t : t + 1], use_cache=True))[:, 0]
             for t in range(x.shape[1])]
    block.reset_cache()

    err = _max_abs_diff(full[0], onp.stack(steps, axis=1)[0])
    assert err < 1e-10, f"RoPE cached decode differs by {err:.3e}"


def test_t5_cross_attention_cache():
    """CrossAttention caches encoder K/V once; decoding must stay exact."""
    onp.random.seed(0)
    model = T5(vocab_size=32, d_model=16, n_heads=4, n_encoder_layers=2,
               n_decoder_layers=2, d_ff=32, n_buckets=8, max_seq_len=16)
    src = onp.array([[1, 5, 9, 2]])
    tgt = onp.array([[0, 7, 3]])

    full = to_numpy(model.forward(src, tgt))

    model.reset_cache()
    enc = model.encode(src)
    steps = []
    for t in range(tgt.shape[1]):
        step = model.decode(tgt[:, t : t + 1], enc, use_cache=True)
        steps.append(to_numpy(step)[:, -1])
    model.reset_cache()

    err = _max_abs_diff(full[0], onp.stack(steps, axis=1)[0])
    assert err < 1e-10, f"T5 cached decode differs by {err:.3e}"


# ---- grouped-query attention ---------------------------------------------------

def test_gqa_shrinks_the_kv_cache():
    """The whole point of GQA: fewer KV heads to store while decoding."""
    mha = MultiHeadAttention(16, 4, causal=True, max_cache_len=8)
    gqa = MultiHeadAttention(16, 4, causal=True, max_cache_len=8, n_kv_heads=1)

    for layer in (mha, gqa):
        layer.reset_cache()
        layer.forward(_rand(1, 3, 16), use_cache=True)

    assert mha._cache_k.shape[1] == 4
    assert gqa._cache_k.shape[1] == 1, "MQA cache should hold a single KV head"
    # and the K/V projections shrink to match
    assert gqa.W_K.W.data.shape[1] == 4
    assert mha.W_K.W.data.shape[1] == 16


def test_gqa_equals_mha_when_kv_heads_repeated():
    """A GQA layer whose KV weights are tiled must equal plain MHA exactly.

    Repeating the single KV head's projection across all 4 query-head slots
    makes GQA mathematically identical to full multi-head attention — a direct
    check that `_repeat_kv` expands into the right head slots.
    """
    gqa = MultiHeadAttention(16, 4, causal=True, n_kv_heads=1)
    mha = MultiHeadAttention(16, 4, causal=True)

    mha.W_Q.W.data = onp.array(to_numpy(gqa.W_Q.W.data))
    mha.W_Q.b.data = onp.array(to_numpy(gqa.W_Q.b.data))
    mha.W_O.W.data = onp.array(to_numpy(gqa.W_O.W.data))
    mha.W_O.b.data = onp.array(to_numpy(gqa.W_O.b.data))
    for src, dst in ((gqa.W_K, mha.W_K), (gqa.W_V, mha.W_V)):
        dst.W.data = onp.tile(onp.array(to_numpy(src.W.data)), (1, 4))
        dst.b.data = onp.tile(onp.array(to_numpy(src.b.data)), 4)

    x = _rand(2, 5, 16)
    err = _max_abs_diff(gqa.forward(x), mha.forward(x))
    assert err < 1e-12, f"GQA with tiled KV differs from MHA by {err:.3e}"


def test_gqa_cache_matches_full_forward():
    onp.random.seed(0)
    model = GPT2(vocab_size=32, d_model=16, n_heads=4, n_layers=2,
                 max_seq_len=16, n_kv_heads=2)
    tokens = onp.array([[3, 14, 7, 1, 9]])
    full = to_numpy(model.forward(tokens))

    model.reset_cache()
    steps = [to_numpy(model.forward(tokens[:, t : t + 1], use_cache=True))[:, -1]
             for t in range(tokens.shape[1])]
    model.reset_cache()

    err = _max_abs_diff(full[0], onp.stack(steps, axis=1)[0])
    assert err < 1e-10, f"GQA cached decode differs by {err:.3e}"


# ---- padding masks ---------------------------------------------------------------

def test_padding_mask_equals_shorter_sequence():
    """Masked padding must not influence the unpadded positions at all.

    Run a length-3 sequence on its own, then the same sequence padded to
    length 5 with the pad positions masked. The first 3 outputs must match
    exactly — that is what a padding mask is *for*.
    """
    layer = MultiHeadAttention(16, 4, causal=True)
    short = _rand(1, 3, 16)
    padded = onp.concatenate([short, _rand(1, 2, 16, seed=9)], axis=1)
    mask = onp.array([[False, False, False, True, True]])

    out_short = to_numpy(layer.forward(short))
    out_padded = to_numpy(layer.forward(padded, key_padding_mask=mask))

    err = _max_abs_diff(out_short, out_padded[:, :3])
    assert err < 1e-12, f"padding leaked into real positions by {err:.3e}"


def test_padding_mask_without_mask_does_leak():
    """Control: without the mask, padding *does* change the output.

    Guards against a mask implementation that is silently a no-op — if this
    ever passes with a tiny difference, the test above proves nothing.
    """
    layer = MultiHeadAttention(16, 4, causal=True)
    short = _rand(1, 3, 16)
    padded = onp.concatenate([short, _rand(1, 2, 16, seed=9)], axis=1)

    out_short = to_numpy(layer.forward(short))
    out_padded = to_numpy(layer.forward(padded))
    # causal attention means later padding cannot affect earlier queries...
    assert _max_abs_diff(out_short, out_padded[:, :3]) < 1e-12

    # ...so use a non-causal layer, where it certainly can
    layer = MultiHeadAttention(16, 4, causal=False)
    out_short = to_numpy(layer.forward(short))
    out_padded = to_numpy(layer.forward(padded))
    assert _max_abs_diff(out_short, out_padded[:, :3]) > 1e-6, (
        "unmasked padding had no effect — the equivalence test is vacuous"
    )

    masked = to_numpy(layer.forward(padded, key_padding_mask=onp.array(
        [[False, False, False, True, True]]
    )))
    err = _max_abs_diff(out_short, masked[:, :3])
    assert err < 1e-12, f"non-causal padding mask leaked by {err:.3e}"


def test_flash_padding_mask_matches_dense():
    """The tiled kernels must apply the padding mask identically."""
    mask = onp.array([[False, False, False, True, True],
                      [False, False, True, True, True]])
    x = _rand(2, 5, 16)

    for causal in (True, False):
        for flash_cls in (FlashAttention, FlashAttention2):
            dense = MultiHeadAttention(16, 4, causal=causal)
            flash = flash_cls(16, 4, causal=causal, block_q=2, block_k=2)
            _copy_params(dense, flash)
            err = _max_abs_diff(
                dense.forward(x, key_padding_mask=mask),
                flash.forward(x, key_padding_mask=mask),
            )
            assert err < 1e-10, (
                f"{flash_cls.__name__}(causal={causal}) padded output differs by {err:.3e}"
            )


# ---- train / eval mode ------------------------------------------------------------

def test_train_eval_mode_propagates():
    """Mode must reach every nested module, including ones inside lists."""
    model = GPT2(vocab_size=16, d_model=8, n_heads=2, n_layers=2, max_seq_len=8)
    assert model.training is True

    model.eval()
    assert all(not m.training for m in model.modules())
    assert model.blocks[1].attn.W_Q.training is False, "did not reach a nested list child"

    model.train()
    assert all(m.training for m in model.modules())


def test_modules_includes_nested_children():
    model = GPT2(vocab_size=16, d_model=8, n_heads=2, n_layers=2, max_seq_len=8)
    found = model.modules()
    assert model in found
    assert model.blocks[0] in found
    assert model.blocks[0].attn in found
    assert model.blocks[0].attn.W_Q in found


# ---- weight tying ------------------------------------------------------------

def test_tied_projection_not_double_counted():
    """The tied head shares Embedding.W, so it must not add a second entry."""
    model = GPT2(vocab_size=32, d_model=16, n_heads=4, n_layers=1, max_seq_len=16)
    params = model.parameters()
    ids = [id(p) for p in params]
    assert len(ids) == len(set(ids)), "parameters() returned a duplicate Parameter"
    assert any(p is model.token_emb.W for p in params), "embedding weight missing"


# ---- checkpoint utilities -----------------------------------------------------

def test_checkpoint_roundtrip_and_average():
    model = GPT2(vocab_size=16, d_model=8, n_heads=2, n_layers=1, max_seq_len=8)
    state_a = get_state(model)

    for p in model.parameters():
        p.data = onp.asarray(to_numpy(p.data)) + 1.0
    state_b = get_state(model)

    set_state(model, state_a)
    assert _max_abs_diff(model.parameters()[0].data, state_a[0]) == 0.0

    averaged = average_states([state_a, state_b])
    expected = (onp.asarray(to_numpy(state_a[0])) + onp.asarray(to_numpy(state_b[0]))) / 2
    assert _max_abs_diff(averaged[0], expected) < 1e-12


# ---- backend agreement ---------------------------------------------------------

_BACKEND_PROBE = """
import numpy as onp
from core.backend import to_numpy
from models.gpt2 import GPT2

onp.random.seed(0)
model = GPT2(vocab_size=32, d_model=16, n_heads=4, n_layers=2, max_seq_len=16)
tokens = onp.array([[3, 14, 7, 1, 9]])
logits = to_numpy(model.forward(tokens))
grad = onp.random.RandomState(1).randn(*logits.shape)
model.backward(grad)
gsum = sum(float(onp.sum(to_numpy(p.grad))) for p in model.parameters())
print(repr((float(onp.sum(logits)), float(onp.sum(logits ** 2)), gsum)))
"""


def _run_probe(env_extra: dict) -> tuple:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), **env_extra}
    result = subprocess.run(
        [sys.executable, "-c", _BACKEND_PROBE],
        capture_output=True, text=True, env=env, cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        raise RuntimeError(f"probe failed:\n{result.stderr}")
    return eval(result.stdout.strip())


def test_jax_backend_matches_numpy():
    """Same model, same seed, both backends at float64 — must agree.

    Skipped when jax is not installed; run as a subprocess because the backend
    is chosen from the environment at import time.
    """
    try:
        import jax  # noqa: F401
    except ImportError:
        print("        (skipped: jax not installed)")
        return

    numpy_result = _run_probe({"TINY_PRE_TRAIN_BACKEND": "numpy"})
    jax_result = _run_probe(
        {"TINY_PRE_TRAIN_BACKEND": "jax", "TINY_PRE_TRAIN_JAX_X64": "1"}
    )

    for name, a, b in zip(
        ("sum(logits)", "sum(logits^2)", "sum(grads)"), numpy_result, jax_result
    ):
        rel = abs(a - b) / max(abs(a) + abs(b), 1e-8)
        assert rel < 1e-8, f"backend mismatch on {name}: numpy={a!r} jax={b!r}"


def test_backend_reports_itself():
    assert BACKEND in ("numpy", "jax")
