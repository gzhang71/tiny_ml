from core.backend import xp as np
from core.module import Optimizer
from core.parameter import Parameter


class AdamW(Optimizer):
    """Adam with *decoupled* weight decay (Loshchilov & Hutter 2019).

    Classic "Adam + L2" adds λθ to the gradient, so the penalty then passes
    through Adam's 1/√v normalization — parameters with large gradient
    variance end up decayed *less*, which is not what a weight penalty is
    supposed to do. AdamW instead applies the decay directly to the parameter,
    outside the adaptive step:

        Adam+L2:  g ← g + λθ;  θ ← θ − lr · m̂/(√v̂ + ε)
        AdamW:    θ ← θ − lr · m̂/(√v̂ + ε) − lr · λ · θ

    This is the optimizer essentially every modern LM is pre-trained with.

    Weight decay should apply to matmul weights but *not* to biases or
    normalization gains — decaying a LayerNorm γ toward zero just shrinks the
    activations it was meant to rescale. Pass `no_decay` explicitly, or build
    both lists with `decay_groups(model)` from this package.
    """

    def __init__(
        self,
        parameters: list[Parameter],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        no_decay: list[Parameter] | None = None,
    ):
        self._params = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        # identity-based: Parameters are unhashable-by-value arrays
        skip = {id(p) for p in (no_decay or [])}
        self._decay = [0.0 if id(p) in skip else 1.0 for p in parameters]
        self._m = [np.zeros_like(p.data) for p in parameters]
        self._v = [np.zeros_like(p.data) for p in parameters]
        self._t = 0

    def step(self):
        self._t += 1
        bc1 = 1.0 - self.beta1 ** self._t
        bc2 = 1.0 - self.beta2 ** self._t
        for i, p in enumerate(self._params):
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * p.grad
            self._v[i] = self.beta2 * self._v[i] + (1.0 - self.beta2) * p.grad ** 2
            m_hat = self._m[i] / bc1
            v_hat = self._v[i] / bc2
            update = m_hat / (np.sqrt(v_hat) + self.eps)
            if self.weight_decay and self._decay[i]:
                update = update + self.weight_decay * p.data
            p.data = p.data - self.lr * update

    def zero_grad(self):
        for p in self._params:
            p.grad = np.zeros_like(p.data)


def decay_groups(model) -> tuple[list[Parameter], list[Parameter]]:
    """Split `model.parameters()` into (decay, no_decay) by rank.

    The standard heuristic: rank ≥ 2 tensors are matmul weights and get decayed;
    rank ≤ 1 tensors are biases, normalization gains, and scalar scales, and do
    not. Use as:

        params = model.parameters()
        decay, no_decay = decay_groups(model)
        opt = AdamW(params, weight_decay=0.1, no_decay=no_decay)
    """
    params = model.parameters()
    decay = [p for p in params if np.ndim(p.data) >= 2]
    no_decay = [p for p in params if np.ndim(p.data) < 2]
    return decay, no_decay
