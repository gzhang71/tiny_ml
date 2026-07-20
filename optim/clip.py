"""Gradient clipping."""
from core.backend import xp as np
from core.parameter import Parameter


def grad_global_norm(parameters: list[Parameter]) -> float:
    """L2 norm of all gradients concatenated into one vector."""
    total = 0.0
    for p in parameters:
        total += float(np.sum(p.grad * p.grad))
    return float(np.sqrt(total))


def clip_grad_norm(parameters: list[Parameter], max_norm: float) -> float:
    """Rescale all gradients so their *global* L2 norm is at most `max_norm`.

    Returns the norm measured *before* clipping — log it: a run whose grad norm
    suddenly spikes by orders of magnitude is diverging even if the loss has
    not caught up yet.

    Scaling every gradient by one shared factor preserves the update direction
    exactly and only shortens it. Clipping each tensor separately would instead
    rotate the update, which is why the global form is the standard one. In
    transformer pre-training this is what turns a loss spike into a slightly
    slow step rather than a NaN that ends the run.
    """
    total_norm = grad_global_norm(parameters)
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p in parameters:
            p.grad = p.grad * scale  # rebinding, not in-place: jax-safe
    return total_norm
