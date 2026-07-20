"""Learning-rate schedules.

A schedule is just a callable `step -> lr`. Every optimizer in this package
stores its rate in `self.lr`, so applying one is a single assignment:

    schedule = CosineWithWarmup(peak_lr=3e-4, warmup_steps=100, total_steps=5000)
    for step in range(1, total_steps + 1):
        optimizer.lr = schedule(step)
        ...

`Trainer` does this for you when constructed with `lr_schedule=`.

Steps are 1-based, matching the training loops in `examples/`.
"""
import math


class Schedule:
    """Base class: a callable mapping training step → learning rate."""

    def __call__(self, step: int) -> float:
        raise NotImplementedError


class ConstantLR(Schedule):
    """No schedule — useful as an explicit default."""

    def __init__(self, lr: float):
        self.lr = lr

    def __call__(self, step: int) -> float:
        return self.lr


class LinearWarmup(Schedule):
    """Linear ramp from ~0 to `peak_lr` over `warmup_steps`, then constant.

    Warmup exists because Adam's second-moment estimate v is meaningless for
    the first few steps (it is still dominated by its zero initialization), so
    the early adaptive steps are effectively unnormalized and can blow up a
    freshly initialized transformer.
    """

    def __init__(self, peak_lr: float, warmup_steps: int):
        self.peak_lr = peak_lr
        self.warmup_steps = max(1, warmup_steps)

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.peak_lr * step / self.warmup_steps
        return self.peak_lr


class CosineWithWarmup(Schedule):
    """Linear warmup, then cosine decay from `peak_lr` to `min_lr`.

    The default LM pre-training schedule (GPT-3, Chinchilla, Llama). After
    `total_steps` the rate stays pinned at `min_lr` rather than turning back
    up, so overrunning the planned step count degrades gracefully.
    """

    def __init__(self, peak_lr: float, warmup_steps: int, total_steps: int,
                 min_lr: float = 0.0):
        assert total_steps > warmup_steps, "total_steps must exceed warmup_steps"
        self.peak_lr = peak_lr
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = total_steps
        self.min_lr = min_lr

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.peak_lr * step / self.warmup_steps
        if step >= self.total_steps:
            return self.min_lr
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine


class InverseSqrt(Schedule):
    """lr = peak_lr · √(warmup_steps / max(step, warmup_steps)).

    The original Transformer / T5 schedule: warm up linearly, then decay as
    1/√step. Unlike cosine it needs no `total_steps`, so it suits runs whose
    length is not known in advance.
    """

    def __init__(self, peak_lr: float, warmup_steps: int):
        self.peak_lr = peak_lr
        self.warmup_steps = max(1, warmup_steps)

    def __call__(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.peak_lr * step / self.warmup_steps
        return self.peak_lr * math.sqrt(self.warmup_steps / step)
