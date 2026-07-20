"""Finite-difference gradient checks for every layer, head, and loss.

Each test builds a small instance, feeds it a fixed random input, and compares
the hand-derived backward pass against central differences. Sizes are kept tiny
because the numeric check costs 2 forward passes *per parameter entry*.
"""
import numpy as onp

from tests.gradcheck import (
    check_layer,
    check_input_grad,
    check_loss,
    GradCheckError,
    _rel_error,
)

from core.module import Layer
from core.parameter import Parameter
from layers.linear import Linear
from layers.activations import ReLU, Sigmoid, Tanh, SiLU, GeLU
from layers.normalization import LayerNorm, RMSNorm
from layers.embedding import (
    Embedding,
    SinusoidalPositionalEmbedding,
    LearnedPositionalEmbedding,
    FeatureEmbedding,
    RotaryPositionalEmbedding,
)
from layers.feedforward import FeedForward, SwiGLU
from layers.residual import ResidualBlock
from layers.moe import MoEFeedForward
from layers.attention import (
    MultiHeadAttention,
    RoPEAttention,
    T5SelfAttention,
    CrossAttention,
    TransformerBlock,
)
from layers.flash_attention import FlashAttention, FlashAttention2
from layers.dropout import Dropout
from layers.heads import LinearHead, EuclideanHead, CosineHead, HyperbolicHead
from losses.losses import (
    MSELoss,
    SoftmaxCrossEntropy,
    BinaryCrossEntropy,
    BCEWithLogits,
)
from models.mlp import MLP


def _rand(*shape, seed: int = 0, scale: float = 1.0) -> onp.ndarray:
    return onp.random.RandomState(seed).randn(*shape) * scale


# ---- the checker itself ----------------------------------------------------
#
# Everything below trusts `check_layer` to fail when a backward is wrong, so
# that sensitivity has to be demonstrated, not assumed. These two layers are
# deliberately broken in the ways hand-derived gradients actually break.

class _WrongSignLinear(Layer):
    """Linear whose weight gradient has a flipped sign."""

    def __init__(self, in_features: int, out_features: int):
        self.W = Parameter(_rand(in_features, out_features))

    def forward(self, x):
        self._x = x
        return x @ self.W.data

    def backward(self, grad):
        self.W.grad += -(self._x.T @ grad)
        return grad @ self.W.data.T


class _MissingTermNorm(Layer):
    """Mean-centering whose backward forgets the d(mean)/dx contribution.

    The realistic normalization bug: the dominant term is right, so outputs and
    losses look plausible and only the gradient is subtly wrong.
    """

    def __init__(self, d_model: int):
        self.gamma = Parameter(onp.ones(d_model))

    def forward(self, x):
        self._x = x
        self._mu = x.mean(axis=-1, keepdims=True)
        return (x - self._mu) * self.gamma.data

    def backward(self, grad):
        self.gamma.grad += (grad * (self._x - self._mu)).sum(axis=0)
        return grad * self.gamma.data


def test_gradcheck_catches_a_sign_error():
    try:
        check_layer(_WrongSignLinear(3, 2), _rand(4, 3))
    except GradCheckError:
        return
    raise AssertionError("gradient checker missed a flipped-sign weight gradient")


def test_gradcheck_catches_a_missing_term():
    try:
        check_layer(_MissingTermNorm(3), _rand(4, 3))
    except GradCheckError:
        return
    raise AssertionError("gradient checker missed an incomplete input gradient")


# ---- basic layers --------------------------------------------------------

def test_linear():
    check_layer(Linear(4, 3), _rand(5, 4))


def test_linear_3d_input():
    """Linear flattens leading dims in backward — check it on (B, T, D)."""
    check_layer(Linear(4, 3), _rand(2, 3, 4))


def test_activations():
    # offset away from 0 so ReLU's kink never lands inside the +-eps window
    x = _rand(4, 5) + 1.5
    for cls in (ReLU, Sigmoid, Tanh, SiLU, GeLU):
        check_layer(cls(), x, name=cls.__name__)


def test_relu_negative_branch():
    check_layer(ReLU(), _rand(4, 5, seed=3) - 1.5, name="ReLU(negative)")


def test_layernorm():
    check_layer(LayerNorm(6), _rand(4, 6))


def test_rmsnorm():
    check_layer(RMSNorm(6), _rand(4, 6))


def test_norms_3d():
    check_layer(LayerNorm(6), _rand(2, 3, 6), name="LayerNorm(3d)")
    check_layer(RMSNorm(6), _rand(2, 3, 6), name="RMSNorm(3d)")


def test_residual_block():
    check_layer(ResidualBlock(5), _rand(4, 5) + 0.5)


# ---- embeddings ----------------------------------------------------------

def test_embedding_weight_grad():
    """Embedding has no input gradient (integer ids), so params only.

    Repeated ids exercise the scatter-add accumulation path.
    """
    layer = Embedding(7, 4)
    tokens = onp.array([[1, 3, 3], [0, 6, 1]])
    check_layer(layer, tokens, check_input=False)


def test_learned_positional_embedding():
    check_layer(LearnedPositionalEmbedding(8, 5), _rand(2, 4, 5))


def test_sinusoidal_positional_embedding():
    check_layer(SinusoidalPositionalEmbedding(6, max_seq_len=8), _rand(2, 4, 6))


def test_feature_embedding():
    check_layer(FeatureEmbedding(3, 4), _rand(5, 3))


def test_rope_is_orthogonal():
    """RoPE backward is the inverse rotation, so it must preserve norms."""
    rope = RotaryPositionalEmbedding(8, max_seq_len=16)
    x = _rand(2, 3, 5, 8)
    check_input_grad(rope, x, name="RotaryPositionalEmbedding")

    rotated = rope.forward(x, offset=0)
    assert _rel_error(onp.linalg.norm(rotated, axis=-1),
                      onp.linalg.norm(x, axis=-1)) < 1e-12, "RoPE changed norms"


# ---- feed-forward variants ------------------------------------------------

def test_feedforward():
    check_layer(FeedForward(4, 8), _rand(3, 4) + 0.5)


def test_feedforward_gelu():
    check_layer(FeedForward(4, 8, activation_cls=GeLU), _rand(3, 4), name="FeedForward(GeLU)")


def test_swiglu():
    check_layer(SwiGLU(4, 8), _rand(3, 4))


def test_moe_routing():
    """aux_coef=0: backward must be the gradient of the main objective alone."""
    moe = MoEFeedForward(4, 8, n_experts=3, top_k=2, aux_coef=0.0)
    check_layer(moe, _rand(2, 3, 4))


def test_moe_aux_loss_gradient():
    """With aux_coef>0 the router grad carries an extra term that the main
    objective cannot see, so the objective must include `aux_loss` too."""
    moe = MoEFeedForward(4, 8, n_experts=3, top_k=2, aux_coef=0.1)
    x = _rand(2, 3, 4)
    weights = onp.random.RandomState(0).randn(2, 3, 4)

    for p in moe.parameters():
        p.grad = onp.zeros_like(p.data)
    moe.forward(x)
    moe.backward(weights)
    analytic = onp.array(moe.router.W.grad)

    def objective():
        out = moe.forward(x)
        return float(onp.sum(out * weights) + moe.aux_loss)

    eps = 1e-6
    W = moe.router.W
    base = onp.array(W.data)
    numeric = onp.zeros_like(base)
    for idx in onp.ndindex(base.shape):
        original = base[idx]
        base[idx] = original + eps
        W.data = onp.array(base)
        plus = objective()
        base[idx] = original - eps
        W.data = onp.array(base)
        minus = objective()
        base[idx] = original
        numeric[idx] = (plus - minus) / (2 * eps)
    W.data = base

    err = _rel_error(analytic, numeric)
    assert err < 1e-6, f"MoE aux-loss router gradient off by {err:.3e}"


# ---- attention -------------------------------------------------------------

def test_multihead_attention():
    check_layer(MultiHeadAttention(8, 2, causal=True), _rand(2, 4, 8))


def test_multihead_attention_noncausal():
    check_layer(MultiHeadAttention(8, 2, causal=False), _rand(2, 4, 8),
                name="MultiHeadAttention(noncausal)")


def test_rope_attention():
    check_layer(RoPEAttention(8, 2, causal=True), _rand(2, 4, 8))


def test_t5_self_attention():
    """Covers RelativePositionBias.backward (scatter-add over buckets)."""
    check_layer(T5SelfAttention(8, 2, causal=False, n_buckets=8), _rand(2, 4, 8))


def test_flash_attention():
    check_layer(FlashAttention(8, 2, causal=True), _rand(2, 4, 8))


def test_flash_attention_2():
    check_layer(FlashAttention2(8, 2, causal=True), _rand(2, 4, 8))


def test_cross_attention():
    """Two inputs and a tuple gradient, so it needs a bespoke check."""
    layer = CrossAttention(8, 2)
    x_dec, x_enc = _rand(2, 3, 8, seed=1), _rand(2, 5, 8, seed=2)
    weights = onp.random.RandomState(0).randn(2, 3, 8)

    for p in layer.parameters():
        p.grad = onp.zeros_like(p.data)
    layer.forward(x_dec, x_enc)
    d_dec, d_enc = layer.backward(weights)

    eps = 1e-6
    for analytic, arr, label in ((d_dec, x_dec, "x_dec"), (d_enc, x_enc, "x_enc")):
        base = onp.array(arr)
        numeric = onp.zeros_like(base)
        for idx in onp.ndindex(base.shape):
            original = base[idx]
            base[idx] = original + eps
            args = (base, x_enc) if label == "x_dec" else (x_dec, base)
            plus = float(onp.sum(layer.forward(*args) * weights))
            base[idx] = original - eps
            args = (base, x_enc) if label == "x_dec" else (x_dec, base)
            minus = float(onp.sum(layer.forward(*args) * weights))
            base[idx] = original
            numeric[idx] = (plus - minus) / (2 * eps)
        err = _rel_error(analytic, numeric)
        assert err < 1e-6, f"CrossAttention d/d{label} off by {err:.3e}"


def test_grouped_query_attention():
    """n_kv_heads < n_heads: the repeat/sum-back adjoint must be exact."""
    check_layer(MultiHeadAttention(8, 4, causal=True, n_kv_heads=2), _rand(2, 4, 8),
                name="MultiHeadAttention(GQA 4->2)")


def test_multi_query_attention():
    """n_kv_heads=1 is MQA — every query head shares one K/V head."""
    check_layer(MultiHeadAttention(8, 4, causal=True, n_kv_heads=1), _rand(2, 4, 8),
                name="MultiHeadAttention(MQA)")


def test_grouped_query_flash_attention():
    check_layer(FlashAttention2(8, 4, causal=True, n_kv_heads=2), _rand(2, 4, 8),
                name="FlashAttention2(GQA)")


def test_grouped_query_rope_attention():
    """RoPE + GQA is the Llama configuration."""
    check_layer(RoPEAttention(8, 4, causal=True, n_kv_heads=2), _rand(2, 4, 8),
                name="RoPEAttention(GQA)")


def test_attention_with_padding_mask():
    layer = MultiHeadAttention(8, 2, causal=True)
    mask = onp.array([[False, False, True, True], [False, False, False, True]])
    x = _rand(2, 4, 8)

    # bind the mask into forward so the checker's plain forward(x) still applies it
    original = layer.forward
    layer.forward = lambda arr: original(arr, key_padding_mask=mask)
    check_layer(layer, x, name="MultiHeadAttention(padded)")


def test_transformer_block():
    check_layer(TransformerBlock(8, 2, d_ff=16, causal=True), _rand(2, 4, 8))


def test_transformer_block_swiglu():
    check_layer(TransformerBlock(8, 2, d_ff=16, causal=True, ffn_cls=SwiGLU),
                _rand(2, 4, 8), name="TransformerBlock(SwiGLU)")


def test_transformer_block_flash():
    check_layer(TransformerBlock(8, 2, d_ff=16, causal=True, attention_cls=FlashAttention2),
                _rand(2, 4, 8), name="TransformerBlock(Flash2)")


# ---- dropout ----------------------------------------------------------------

def test_dropout_eval_is_identity():
    """In eval mode dropout must be exactly the identity, forward and back."""
    layer = Dropout(0.5).eval()
    x = _rand(4, 6)
    assert onp.array_equal(layer.forward(x), x)
    grad = _rand(4, 6, seed=1)
    assert onp.array_equal(layer.backward(grad), grad)
    check_layer(layer, x, name="Dropout(eval)")


def test_dropout_train_masks_and_rescales():
    onp.random.seed(0)
    layer = Dropout(0.5).train()
    x = onp.ones((200, 200))
    out = layer.forward(x)

    dropped = onp.mean(out == 0.0)
    assert 0.45 < dropped < 0.55, f"dropped {dropped:.3f}, expected ~0.5"
    # inverted dropout: surviving units are scaled by 1/(1-p) so the mean holds
    assert abs(out.mean() - 1.0) < 0.02, f"mean {out.mean():.4f} should stay ~1.0"
    assert onp.allclose(out[out != 0.0], 2.0), "kept units must scale by 1/(1-p)"


def test_dropout_backward_uses_forward_mask():
    """The backward mask must be the one drawn in forward, not a fresh draw."""
    onp.random.seed(0)
    layer = Dropout(0.5).train()
    x = _rand(20, 20)
    out = layer.forward(x)
    grad = layer.backward(onp.ones_like(x))
    assert onp.array_equal(out == 0.0, grad == 0.0), "backward dropped different units"


def test_dropout_zero_p_is_identity_in_training():
    layer = Dropout(0.0).train()
    x = _rand(4, 6)
    assert onp.array_equal(layer.forward(x), x)
    check_layer(layer, x, name="Dropout(p=0)")


# ---- heads -----------------------------------------------------------------

def test_linear_head():
    check_layer(LinearHead(4, 3), _rand(5, 4))


def test_euclidean_head():
    check_layer(EuclideanHead(4, 3), _rand(5, 4))


def test_cosine_head():
    """Includes the 0-d learnable scale Parameter."""
    check_layer(CosineHead(4, 3), _rand(5, 4))


def test_hyperbolic_head():
    # modest feature norms keep points well inside the ball
    check_layer(HyperbolicHead(4, 3, curvature=1.0), _rand(5, 4, scale=0.3))


def test_hyperbolic_head_near_origin():
    """Exercises the r->0 series branch of _expmap0_backward."""
    check_layer(HyperbolicHead(4, 3, curvature=1.0), _rand(5, 4, scale=1e-4),
                tol=1e-4, name="HyperbolicHead(near origin)")


# ---- models ----------------------------------------------------------------

def test_mlp():
    check_layer(MLP([4, 6, 3]), _rand(5, 4) + 0.5)


# ---- losses ----------------------------------------------------------------

def test_mse_loss():
    check_loss(MSELoss(), _rand(4, 3), _rand(4, 3, seed=1))


def test_softmax_cross_entropy_integer_labels():
    check_loss(SoftmaxCrossEntropy(), _rand(5, 4), onp.array([0, 3, 1, 1, 2]))


def test_softmax_cross_entropy_one_hot():
    targets = onp.eye(4)[onp.array([0, 3, 1, 1, 2])]
    check_loss(SoftmaxCrossEntropy(), _rand(5, 4), targets,
               name="SoftmaxCrossEntropy(one-hot)")


def test_softmax_cross_entropy_sequence_labels():
    """(B, T, C) logits with (B, T) integer labels — the language-model case."""
    labels = onp.array([[0, 3, 1], [2, 2, 0]])
    check_loss(SoftmaxCrossEntropy(), _rand(2, 3, 4), labels,
               name="SoftmaxCrossEntropy(B,T)")


def test_softmax_cross_entropy_flattening_is_equivalent():
    """Averaging must be over B·T, so flattening first changes nothing."""
    logits, labels = _rand(2, 3, 4), onp.array([[0, 3, 1], [2, 2, 0]])
    nested = SoftmaxCrossEntropy().forward(logits, labels)
    flat = SoftmaxCrossEntropy().forward(logits.reshape(-1, 4), labels.reshape(-1))
    assert abs(nested - flat) < 1e-12, f"{nested} != {flat}"


def test_softmax_cross_entropy_rejects_ambiguous_targets():
    """The footgun this shape check exists for: (B,T) labels read as one-hot."""
    loss = SoftmaxCrossEntropy()
    try:
        loss.forward(_rand(2, 3, 4), onp.zeros((2, 5)))
    except ValueError as exc:
        assert "matches neither" in str(exc)
    else:
        raise AssertionError("mismatched target shape should raise")


def test_binary_cross_entropy():
    # inputs must be probabilities; keep them away from the 0/1 clip boundaries
    pred = 1.0 / (1.0 + onp.exp(-_rand(4, 3)))
    target = (onp.random.RandomState(1).rand(4, 3) > 0.5).astype(float)
    check_loss(BinaryCrossEntropy(), pred, target)


def test_bce_with_logits():
    target = (onp.random.RandomState(1).rand(4, 3) > 0.5).astype(float)
    check_loss(BCEWithLogits(), _rand(4, 3), target)


def test_bce_with_logits_matches_unfused_in_the_easy_range():
    """Same function as sigmoid + BinaryCrossEntropy where that one is safe."""
    logits = _rand(4, 3)
    target = (onp.random.RandomState(1).rand(4, 3) > 0.5).astype(float)
    probs = 1.0 / (1.0 + onp.exp(-logits))

    fused = BCEWithLogits().forward(logits, target)
    unfused = BinaryCrossEntropy().forward(probs, target)
    assert abs(fused - unfused) < 1e-12, f"{fused} != {unfused}"


def test_bce_with_logits_survives_saturation():
    """Large-magnitude logits: the unfused path clips, the fused one does not."""
    logits = onp.array([[-800.0, 800.0, 50.0]])
    target = onp.array([[0.0, 1.0, 1.0]])

    loss = BCEWithLogits().forward(logits, target)
    assert onp.isfinite(loss), "fused BCE overflowed"
    assert loss < 1e-10, f"confident correct predictions should cost ~0, got {loss}"

    # gradients stay exact well into the range where the unfused form clips
    check_loss(BCEWithLogits(), onp.array([[-30.0, 30.0, 5.0]]),
               onp.array([[0.0, 1.0, 1.0]]), name="BCEWithLogits(saturated)")
