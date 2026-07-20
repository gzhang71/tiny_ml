"""Tests for optimizers, schedules, clipping, checkpointing, and the Trainer."""
import os
import tempfile

import numpy as onp

from core.backend import to_numpy
from core.parameter import Parameter
from losses.losses import MSELoss
from models.mlp import MLP
from optim.sgd import SGD
from optim.adam import ADAM
from optim.adamw import AdamW, decay_groups
from optim.clip import clip_grad_norm, grad_global_norm
from optim.schedule import ConstantLR, LinearWarmup, CosineWithWarmup, InverseSqrt
from training.trainer import Trainer
from training import checkpoint


def _rand(*shape, seed: int = 0) -> onp.ndarray:
    return onp.random.RandomState(seed).randn(*shape)


def _param(values) -> Parameter:
    p = Parameter(onp.array(values, dtype=float))
    p.grad = onp.zeros_like(p.data)
    return p


# ---- AdamW ------------------------------------------------------------------

def test_adamw_decay_is_decoupled():
    """With zero gradient, the only movement must be exactly lr·wd·θ.

    This is the property that distinguishes AdamW from Adam+L2: an L2 penalty
    would enter through the gradient and get divided by √v̂, making the actual
    shrinkage depend on gradient history.
    """
    p = _param([1.0, -2.0, 4.0])
    opt = AdamW([p], lr=0.1, weight_decay=0.5)
    opt.step()

    expected = onp.array([1.0, -2.0, 4.0]) * (1.0 - 0.1 * 0.5)
    assert onp.allclose(to_numpy(p.data), expected), to_numpy(p.data)


def test_adamw_no_decay_list_is_respected():
    decayed, kept = _param([1.0]), _param([1.0])
    opt = AdamW([decayed, kept], lr=0.1, weight_decay=0.5, no_decay=[kept])
    opt.step()

    assert to_numpy(decayed.data)[0] < 1.0, "weight should have been decayed"
    assert to_numpy(kept.data)[0] == 1.0, "no_decay parameter must not move"


def test_adamw_matches_adam_when_decay_is_zero():
    p_adam, p_adamw = _param([1.0, 2.0]), _param([1.0, 2.0])
    adam, adamw = ADAM([p_adam], lr=0.1), AdamW([p_adamw], lr=0.1, weight_decay=0.0)

    for _ in range(3):
        p_adam.grad = onp.array([0.5, -1.0])
        p_adamw.grad = onp.array([0.5, -1.0])
        adam.step()
        adamw.step()

    assert onp.allclose(to_numpy(p_adam.data), to_numpy(p_adamw.data))


def test_decay_groups_splits_by_rank():
    model = MLP([4, 5, 3])
    decay, no_decay = decay_groups(model)

    assert all(onp.ndim(to_numpy(p.data)) >= 2 for p in decay), "non-matrix in decay set"
    assert all(onp.ndim(to_numpy(p.data)) < 2 for p in no_decay), "matrix in no_decay set"
    assert len(decay) + len(no_decay) == len(model.parameters())
    assert len(no_decay) > 0, "biases should land in the no-decay group"


# ---- gradient clipping --------------------------------------------------------

def test_clip_grad_norm_scales_to_max():
    a, b = _param([3.0, 0.0]), _param([4.0])
    a.grad, b.grad = onp.array([3.0, 0.0]), onp.array([4.0])
    assert abs(grad_global_norm([a, b]) - 5.0) < 1e-12

    returned = clip_grad_norm([a, b], max_norm=1.0)
    assert abs(returned - 5.0) < 1e-12, "must return the pre-clip norm"
    assert abs(grad_global_norm([a, b]) - 1.0) < 1e-5


def test_clip_grad_norm_preserves_direction():
    p = _param([0.0, 0.0, 0.0])
    p.grad = onp.array([1.0, 2.0, -3.0])
    before = onp.array(p.grad) / onp.linalg.norm(p.grad)

    clip_grad_norm([p], max_norm=0.5)
    after = onp.array(to_numpy(p.grad)) / onp.linalg.norm(to_numpy(p.grad))
    assert onp.allclose(before, after, atol=1e-9), "clipping rotated the update"


def test_clip_grad_norm_is_noop_below_threshold():
    p = _param([0.0])
    p.grad = onp.array([0.3])
    clip_grad_norm([p], max_norm=10.0)
    assert to_numpy(p.grad)[0] == 0.3, "gradient under the threshold was modified"


# ---- schedules ------------------------------------------------------------------

def test_constant_lr():
    schedule = ConstantLR(3e-4)
    assert schedule(1) == schedule(1000) == 3e-4


def test_linear_warmup():
    schedule = LinearWarmup(peak_lr=1.0, warmup_steps=10)
    assert abs(schedule(5) - 0.5) < 1e-12
    assert schedule(10) == 1.0
    assert schedule(500) == 1.0, "should hold at peak after warmup"


def test_cosine_with_warmup():
    schedule = CosineWithWarmup(peak_lr=1.0, warmup_steps=10, total_steps=110,
                                min_lr=0.1)
    assert abs(schedule(5) - 0.5) < 1e-12, "linear ramp during warmup"
    assert abs(schedule(10) - 1.0) < 1e-12, "peak at end of warmup"
    assert abs(schedule(60) - 0.55) < 1e-9, "halfway should be the cosine midpoint"
    assert abs(schedule(110) - 0.1) < 1e-12, "ends at min_lr"
    assert schedule(5000) == 0.1, "must stay pinned at min_lr past total_steps"


def test_cosine_is_monotonically_decreasing_after_warmup():
    schedule = CosineWithWarmup(peak_lr=1.0, warmup_steps=5, total_steps=100)
    values = [schedule(s) for s in range(5, 101)]
    assert all(b <= a + 1e-15 for a, b in zip(values, values[1:])), "cosine went back up"


def test_inverse_sqrt():
    schedule = InverseSqrt(peak_lr=1.0, warmup_steps=100)
    assert abs(schedule(50) - 0.5) < 1e-12
    assert abs(schedule(100) - 1.0) < 1e-12
    assert abs(schedule(400) - 0.5) < 1e-12, "1/sqrt decay: 4x steps -> half the rate"


# ---- Trainer ----------------------------------------------------------------------

def test_gradient_accumulation_matches_one_large_batch():
    """Two micro-batches of 4 must produce the same gradient as one batch of 8.

    This is what the 1/k loss scaling in `_accumulate` buys; without it the
    accumulated gradient would be k times too large.
    """
    x, y = _rand(8, 3), _rand(8, 2, seed=1)

    onp.random.seed(0)
    model_full = MLP([3, 5, 2])
    trainer_full = Trainer(model_full, MSELoss(), SGD(model_full.parameters(), lr=0.0))
    trainer_full.train_step(x, y)
    grads_full = [onp.array(to_numpy(p.grad)) for p in model_full.parameters()]

    onp.random.seed(0)
    model_accum = MLP([3, 5, 2])
    trainer_accum = Trainer(model_accum, MSELoss(),
                            SGD(model_accum.parameters(), lr=0.0),
                            grad_accum_steps=2)
    trainer_accum.train_step_accumulated([(x[:4], y[:4]), (x[4:], y[4:])])
    grads_accum = [onp.array(to_numpy(p.grad)) for p in model_accum.parameters()]

    for i, (a, b) in enumerate(zip(grads_full, grads_accum)):
        err = float(onp.max(onp.abs(a - b)))
        assert err < 1e-12, f"accumulated gradient {i} differs by {err:.3e}"


def test_trainer_applies_lr_schedule():
    model = MLP([3, 4, 2])
    optimizer = SGD(model.parameters(), lr=99.0)
    schedule = LinearWarmup(peak_lr=1.0, warmup_steps=4)
    trainer = Trainer(model, MSELoss(), optimizer, lr_schedule=schedule)

    trainer.train_step(_rand(4, 3), _rand(4, 2, seed=1))
    assert optimizer.lr == schedule(1), "LR not taken from the schedule"
    assert trainer.step == 1


def test_trainer_counts_optimizer_steps_not_microbatches():
    model = MLP([3, 4, 2])
    trainer = Trainer(model, MSELoss(), SGD(model.parameters(), lr=0.0),
                      grad_accum_steps=2)
    trainer.train_step_accumulated([(_rand(2, 3), _rand(2, 2, seed=1))] * 2)
    assert trainer.step == 1, "accumulation must not inflate the step counter"


def test_trainer_clips_gradients():
    model = MLP([3, 4, 2])
    trainer = Trainer(model, MSELoss(), SGD(model.parameters(), lr=0.0),
                      max_grad_norm=1e-4)
    trainer.train_step(_rand(16, 3) * 100.0, _rand(16, 2, seed=1) * 100.0)

    assert trainer.last_grad_norm > 1e-4, "test data should have produced a big norm"
    assert grad_global_norm(model.parameters()) <= 1e-4 + 1e-9


def test_trainer_fit_reports_validation_loss():
    onp.random.seed(0)
    model = MLP([3, 8, 2])
    trainer = Trainer(model, MSELoss(), SGD(model.parameters(), lr=0.01))
    history = trainer.fit(_rand(32, 3), _rand(32, 2, seed=1), epochs=3,
                          batch_size=8, verbose=False,
                          val_data=(_rand(16, 3, seed=2), _rand(16, 2, seed=3)))

    assert list(history) == ["loss", "val_loss"]
    assert len(history["loss"]) == len(history["val_loss"]) == 3


def test_trainer_fit_reduces_loss():
    """End-to-end sanity: the loop must actually learn something."""
    onp.random.seed(0)
    x = _rand(64, 3)
    y = x @ onp.array([[1.0, -2.0], [0.5, 0.5], [-1.0, 1.0]])

    model = MLP([3, 16, 2])
    trainer = Trainer(model, MSELoss(), ADAM(model.parameters(), lr=0.05))
    history = trainer.fit(x, y, epochs=20, batch_size=16, verbose=False)

    assert history["loss"][-1] < 0.25 * history["loss"][0], (
        f"loss barely moved: {history['loss'][0]:.4f} -> {history['loss'][-1]:.4f}"
    )


def test_trainer_predict_runs_in_eval_mode():
    """predict() must disable dropout and restore the previous mode."""
    from layers.dropout import Dropout
    from models.sequential import Sequential
    from layers.linear import Linear

    model = Sequential([Linear(4, 4), Dropout(0.9)])
    trainer = Trainer(model, MSELoss(), SGD(model.parameters(), lr=0.0))
    model.train()

    x = onp.ones((8, 4))
    a = to_numpy(trainer.predict(x))
    b = to_numpy(trainer.predict(x))
    assert onp.array_equal(a, b), "predict() was stochastic — dropout stayed on"
    assert model.training is True, "predict() did not restore training mode"


# ---- checkpointing --------------------------------------------------------------

def test_checkpoint_save_load_roundtrip():
    onp.random.seed(0)
    model = MLP([3, 5, 2])
    original = [onp.array(to_numpy(p.data)) for p in model.parameters()]

    with tempfile.TemporaryDirectory() as tmp:
        path = checkpoint.save(os.path.join(tmp, "ckpt"), model, step=42, loss=1.5)
        assert path.endswith(".npz")

        for p in model.parameters():  # scribble over the weights
            p.data = onp.zeros_like(to_numpy(p.data))

        meta = checkpoint.load(path, model)
        assert meta["step"] == 42
        assert abs(meta["loss"] - 1.5) < 1e-12
        for before, p in zip(original, model.parameters()):
            assert onp.allclose(before, to_numpy(p.data))


def test_checkpoint_load_rejects_wrong_architecture():
    with tempfile.TemporaryDirectory() as tmp:
        path = checkpoint.save(os.path.join(tmp, "ckpt"), MLP([3, 5, 2]))

        try:
            checkpoint.load(path, MLP([3, 9, 2]))
        except ValueError as exc:
            assert "shape mismatch" in str(exc) or "parameters" in str(exc)
        else:
            raise AssertionError("loading a mismatched checkpoint should raise")


def test_checkpoint_load_rejects_wrong_parameter_count():
    with tempfile.TemporaryDirectory() as tmp:
        path = checkpoint.save(os.path.join(tmp, "ckpt"), MLP([3, 5, 2]))

        try:
            checkpoint.load(path, MLP([3, 5, 5, 2]))
        except ValueError as exc:
            assert "parameters" in str(exc)
        else:
            raise AssertionError("parameter-count mismatch should raise")
