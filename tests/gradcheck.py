"""Finite-difference gradient checking.

Every backward pass in this library is derived by hand, so the one thing worth
testing above all else is that `backward` actually computes the derivative of
`forward`. This module does that numerically: it perturbs each entry of an
array by ±eps, measures how a scalar objective responds, and compares the
resulting central difference

    dL/dθ_i ≈ (L(θ + eps·e_i) − L(θ − eps·e_i)) / (2·eps)

against the analytic gradient the layer produced. Central differences have
O(eps²) truncation error, so at float64 with eps=1e-6 the numeric value is good
to roughly 1e-10 — far tighter than any real bug.

Two entry points:

    check_input_grad(layer, x)   — verifies the value returned by `backward`
    check_param_grads(layer, x)  — verifies `Parameter.grad` accumulation

Both reduce the layer output to a scalar with a *fixed random* weighting
`sum(w ∘ forward(x))`, which makes the check cover every output element (a
plain `sum` would miss errors that cancel across outputs) while keeping the
objective scalar so finite differences apply.

Requires float64. In jax mode run with TINY_PRE_TRAIN_JAX_X64=1 (the default);
float32 has nowhere near the precision for a 1e-6 perturbation.
"""
import numpy as onp

from core.backend import to_numpy
from core.parameter import Parameter


class GradCheckError(AssertionError):
    """Raised when an analytic gradient disagrees with the numeric one."""


def _as_input(x) -> onp.ndarray:
    """Numpy view of a layer input, promoted to float64 only if it is real-valued.

    Integer inputs (token ids) must keep their dtype — they index into an
    embedding table and are not differentiable anyway.
    """
    arr = onp.asarray(to_numpy(x))
    return arr.astype(onp.float64) if onp.issubdtype(arr.dtype, onp.floating) else arr


# Central differences at eps=1e-6 evaluate a float64 objective of order 1-10,
# so the objective carries ~1e-15 of roundoff and the quotient carries ~1e-9.
# Any comparison must therefore allow an absolute slack around that floor, or
# every genuinely-zero gradient reads as a 100% relative error.
_ATOL = 1e-7


def _rel_error(a: onp.ndarray, b: onp.ndarray) -> float:
    """Worst-entry relative error, ignoring differences below the noise floor."""
    a, b = onp.asarray(a, dtype=onp.float64), onp.asarray(b, dtype=onp.float64)
    diff = onp.abs(a - b)
    scale = onp.abs(a) + onp.abs(b)
    # entries explained by finite-difference roundoff alone score 0
    significant = onp.maximum(diff - _ATOL, 0.0)
    return float(onp.max(significant / onp.maximum(scale, _ATOL)))


def _objective_fn(forward, weights):
    """Scalar objective sum(w ∘ forward(*args)) — the L in dL/dθ."""
    def objective(*args):
        return float(onp.sum(to_numpy(forward(*args)) * weights))
    return objective


def _numeric_grad(objective, arr: onp.ndarray, eps: float, set_value) -> onp.ndarray:
    """Central-difference gradient of `objective` w.r.t. every entry of `arr`.

    `set_value(new_arr)` installs a candidate array wherever the layer reads it
    from; this indirection is what lets the same routine check both an input
    array and a Parameter's `.data`.
    """
    base = onp.array(to_numpy(arr), dtype=onp.float64)
    grad = onp.zeros_like(base)
    for idx in onp.ndindex(base.shape):
        original = base[idx]

        base[idx] = original + eps
        set_value(onp.array(base))
        plus = objective()

        base[idx] = original - eps
        set_value(onp.array(base))
        minus = objective()

        base[idx] = original
        grad[idx] = (plus - minus) / (2.0 * eps)

    set_value(onp.array(base))  # restore
    return grad


def check_input_grad(layer, x, *, eps: float = 1e-6, tol: float = 1e-6,
                     seed: int = 0, name: str = "") -> float:
    """Check that `layer.backward(grad)` returns the true dL/dx.

    Returns the relative error; raises GradCheckError if it exceeds `tol`.
    """
    x = _as_input(x)

    rng = onp.random.RandomState(seed)
    weights = rng.randn(*onp.shape(to_numpy(layer.forward(x))))

    # analytic: seed backward with dL/d(out) = w, since L = sum(w ∘ out)
    _zero_grads(layer)
    layer.forward(x)
    analytic = to_numpy(layer.backward(weights))

    holder = {"x": x}
    objective = lambda: float(
        onp.sum(to_numpy(layer.forward(holder["x"])) * weights)
    )
    numeric = _numeric_grad(objective, x, eps, lambda v: holder.__setitem__("x", v))

    err = _rel_error(analytic, numeric)
    if err > tol:
        raise GradCheckError(
            f"{name or type(layer).__name__}: d/dx relative error {err:.3e} > {tol:.1e}\n"
            f"  analytic[:4] = {onp.asarray(analytic).ravel()[:4]}\n"
            f"  numeric [:4] = {numeric.ravel()[:4]}"
        )
    return err


def check_param_grads(layer, x, *, eps: float = 1e-6, tol: float = 1e-6,
                      seed: int = 0, name: str = "") -> dict[int, float]:
    """Check every Parameter's accumulated `.grad` against finite differences.

    Returns {parameter index: relative error}; raises on the first failure.
    """
    x = _as_input(x)
    params = layer.parameters()
    if not params:
        return {}

    rng = onp.random.RandomState(seed)
    weights = rng.randn(*onp.shape(to_numpy(layer.forward(x))))

    _zero_grads(layer)
    layer.forward(x)
    layer.backward(weights)
    analytic = [onp.array(to_numpy(p.grad)) for p in params]

    errors = {}
    for i, p in enumerate(params):
        objective = lambda: float(onp.sum(to_numpy(layer.forward(x)) * weights))
        numeric = _numeric_grad(
            objective, p.data, eps, lambda v: setattr(p, "data", v)
        )
        err = _rel_error(analytic[i], numeric)
        errors[i] = err
        if err > tol:
            raise GradCheckError(
                f"{name or type(layer).__name__}: param[{i}] shape "
                f"{onp.shape(to_numpy(p.data))} relative error {err:.3e} > {tol:.1e}\n"
                f"  analytic[:4] = {analytic[i].ravel()[:4]}\n"
                f"  numeric [:4] = {numeric.ravel()[:4]}"
            )
    return errors


def check_layer(layer, x, *, eps: float = 1e-6, tol: float = 1e-6,
                seed: int = 0, name: str = "", check_input: bool = True):
    """Run both checks on a layer. The usual one-call entry point."""
    name = name or type(layer).__name__
    if check_input:
        check_input_grad(layer, x, eps=eps, tol=tol, seed=seed, name=name)
    check_param_grads(layer, x, eps=eps, tol=tol, seed=seed, name=name)


def check_loss(loss_fn, pred, target, *, eps: float = 1e-6, tol: float = 1e-6,
               name: str = "") -> float:
    """Check that `loss.backward()` returns d(loss)/d(pred).

    Losses already produce a scalar, so no random weighting is needed.
    """
    pred = onp.array(to_numpy(pred), dtype=onp.float64)

    loss_fn.forward(pred, target)
    analytic = to_numpy(loss_fn.backward())

    objective_holder = {"p": pred}
    objective = lambda: float(loss_fn.forward(objective_holder["p"], target))
    numeric = _numeric_grad(
        objective, pred, eps, lambda v: objective_holder.__setitem__("p", v)
    )

    err = _rel_error(analytic, numeric)
    if err > tol:
        raise GradCheckError(
            f"{name or type(loss_fn).__name__}: d/d(pred) relative error "
            f"{err:.3e} > {tol:.1e}\n"
            f"  analytic[:4] = {onp.asarray(analytic).ravel()[:4]}\n"
            f"  numeric [:4] = {numeric.ravel()[:4]}"
        )
    return err


def _zero_grads(layer) -> None:
    for p in layer.parameters():
        p.grad = onp.zeros_like(onp.asarray(to_numpy(p.data), dtype=onp.float64))


__all__ = [
    "GradCheckError",
    "check_input_grad",
    "check_param_grads",
    "check_layer",
    "check_loss",
]
