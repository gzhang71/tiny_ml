"""Classification heads: features (..., in_features) → logits (..., n_classes).

`Head` owns the interface and the leading-dim flattening; subclasses supply
the scoring geometry by overriding `_logits`/`_logits_backward` on 2-D arrays
(the same hook pattern `Attention` uses). Every head produces logits that feed
`SoftmaxCrossEntropy` directly:

- `LinearHead`      — affine:      logits = x @ W + b (wraps `Linear`)
- `EuclideanHead`   — prototypes:  logits = −‖x − p_k‖²        (Snell et al. 2017)
- `CosineHead`      — spherical:   logits = s · cos(x, w_k)    (NormFace / cosine classifier)
- `HyperbolicHead`  — Poincaré:    logits = −d_c(exp₀x, exp₀p) (Ganea et al. 2018)
"""
from core.backend import xp as np, randn
from core.module import Layer
from core.parameter import Parameter
from layers.linear import Linear

_EPS = 1e-12


class Head(Layer):
    """Base classification head.

    forward(x: (..., in_features)) → (..., n_classes)

    Subclasses implement `_logits(x2d)` and `_logits_backward(grad2d)` on
    (N, in_features) / (N, n_classes) arrays and follow the usual layer
    contract (save state on forward, accumulate parameter grads on backward).
    """

    def __init__(self, in_features: int, n_classes: int):
        self.in_features = in_features
        self.n_classes = n_classes
        self._shape: tuple | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._shape = x.shape
        logits = self._logits(x.reshape(-1, x.shape[-1]))
        return logits.reshape(*x.shape[:-1], self.n_classes)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        dx = self._logits_backward(grad.reshape(-1, grad.shape[-1]))
        return dx.reshape(self._shape)

    def _logits(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _logits_backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LinearHead(Head):
    """Standard affine head — `Linear` under the common head interface."""

    def __init__(self, in_features: int, n_classes: int):
        super().__init__(in_features, n_classes)
        self.proj = Linear(in_features, n_classes)

    def _logits(self, x: np.ndarray) -> np.ndarray:
        return self.proj.forward(x)

    def _logits_backward(self, grad: np.ndarray) -> np.ndarray:
        return self.proj.backward(grad)


class EuclideanHead(Head):
    """Prototypical head: logits = −‖x − p_k‖² to learnable class prototypes.

    Squared distance keeps the gradient finite when a feature lands on a
    prototype (plain distance has a 1/d singularity there).
    """

    def __init__(self, in_features: int, n_classes: int):
        super().__init__(in_features, n_classes)
        self.P = Parameter(randn(n_classes, in_features) * 0.01)
        self._x: np.ndarray | None = None

    def _logits(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        P = self.P.data
        return -(np.sum(x * x, axis=-1)[:, None] - 2.0 * x @ P.T + np.sum(P * P, axis=-1)[None, :])

    def _logits_backward(self, grad: np.ndarray) -> np.ndarray:
        x, P = self._x, self.P.data
        self.P.grad += 2.0 * (grad.T @ x - grad.sum(axis=0)[:, None] * P)
        return 2.0 * (grad @ P - grad.sum(axis=1, keepdims=True) * x)


class CosineHead(Head):
    """Cosine classifier: logits = s · cos(x, w_k) with a learnable scale s.

    Both features and class weights are L2-normalized, so logits depend only
    on direction; s (a 0-d Parameter, shared across classes) recovers the
    dynamic range softmax needs. The margin variants (ArcFace/CosFace) modify
    this score using the target label, which the layer API doesn't see — apply
    such margins in the loss if needed.
    """

    def __init__(self, in_features: int, n_classes: int, scale: float = 10.0):
        super().__init__(in_features, n_classes)
        self.W = Parameter(randn(n_classes, in_features) * np.sqrt(1.0 / in_features))
        self.s = Parameter(np.asarray(float(scale)))

    def _logits(self, x: np.ndarray) -> np.ndarray:
        self._xn = np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), _EPS))
        self._wn = np.sqrt(np.maximum(np.sum(self.W.data ** 2, axis=-1, keepdims=True), _EPS))
        self._u = x / self._xn            # (N, D) unit features
        self._v = self.W.data / self._wn  # (K, D) unit class weights
        self._cos = self._u @ self._v.T
        return self.s.data * self._cos

    def _logits_backward(self, grad: np.ndarray) -> np.ndarray:
        u, v = self._u, self._v
        self.s.grad += np.sum(grad * self._cos)
        gc = self.s.data * grad
        du = gc @ v
        dv = gc.T @ u
        # through L2 normalization: d(a/‖a‖) projects out the radial component
        self.W.grad += (dv - np.sum(dv * v, axis=-1, keepdims=True) * v) / self._wn
        return (du - np.sum(du * u, axis=-1, keepdims=True) * u) / self._xn


def _expmap0(v: np.ndarray, c: float):
    """Exponential map at the origin of the Poincaré ball with curvature -c.

    Maps a Euclidean (tangent) vector v into the open ball of radius 1/√c:
    exp₀(v) = tanh(√c ‖v‖) · v / (√c ‖v‖). tanh < 1 keeps the image strictly
    inside the ball, so no projection/clipping step is needed.

    Returns (z, r, f) where z = f·v, r = ‖v‖ and f = tanh(√c r)/(√c r) —
    r and f are saved for the backward pass.
    """
    a = np.sqrt(c)
    r = np.sqrt(np.maximum(np.sum(v * v, axis=-1, keepdims=True), _EPS))
    f = np.tanh(a * r) / (a * r)
    return f * v, r, f


def _expmap0_backward(g: np.ndarray, v: np.ndarray, r: np.ndarray, f: np.ndarray, c: float) -> np.ndarray:
    """d_loss/dv given g = d_loss/dz for z = exp₀(v) = f(r)·v.

    dz/dv = f·I + (f'(r)/r) · v vᵀ. Below the series threshold the closed form
    for f'(r)/r = c(1 − tanh² − f)/(√c r)² loses all precision to cancellation
    (three ≈1 terms differing at O(r²)), so the r→0 limit −2c/3 is used instead.
    """
    ar = np.sqrt(c) * r
    t = np.tanh(ar)
    h = np.where(ar > 1e-3, c * (1.0 - t * t - f) / (ar * ar), -2.0 * c / 3.0)
    return f * g + h * np.sum(v * g, axis=-1, keepdims=True) * v


class HyperbolicHead(Head):
    """Hyperbolic head on the Poincaré ball (curvature -c).

    Lifts features onto the ball with exp₀ and scores each class by the
    negative geodesic distance to a learnable class prototype
    (Ganea et al. 2018; Khrulkov et al., "Hyperbolic Image Embeddings"):

        logits[i, k] = −d_c(exp₀(x_i), exp₀(P_k))
        d_c(u, v)    = arccosh(1 + 2c‖u−v‖² / ((1−c‖u‖²)(1−c‖v‖²))) / √c

    Prototypes are stored as *tangent-space* vectors P and mapped through exp₀
    every forward, so the Euclidean optimizers in `optim/` apply unchanged —
    no Riemannian SGD, and no parameter ever leaves the ball.
    """

    def __init__(self, in_features: int, n_classes: int, curvature: float = 1.0):
        super().__init__(in_features, n_classes)
        self.P = Parameter(randn(n_classes, in_features) * 0.01)
        self.c = curvature
        self._x: np.ndarray | None = None

    def _logits(self, x: np.ndarray) -> np.ndarray:
        c = self.c
        self._x = x
        z, self._r_x, self._f_x = _expmap0(x, c)
        p, self._r_p, self._f_p = _expmap0(self.P.data, c)

        # α/β are strictly positive inside the ball; the clamp only guards
        # float saturation of tanh at extreme feature norms
        alpha = np.maximum(1.0 - c * np.sum(z * z, axis=-1, keepdims=True), _EPS)  # (N, 1)
        beta = np.maximum(1.0 - c * np.sum(p * p, axis=-1), _EPS)[None, :]         # (1, K)
        sq = np.maximum(
            np.sum(z * z, axis=-1)[:, None] - 2.0 * (z @ p.T) + np.sum(p * p, axis=-1)[None, :],
            0.0,
        )
        delta = 2.0 * c * sq / (alpha * beta)

        self._z, self._p = z, p
        self._alpha, self._beta, self._delta = alpha, beta, delta
        return -np.arccosh(1.0 + delta) / np.sqrt(c)

    def _logits_backward(self, grad: np.ndarray) -> np.ndarray:
        c = self.c
        z, p = self._z, self._p
        alpha, beta, delta = self._alpha, self._beta, self._delta

        # d = arccosh(1+δ)/√c, logits = −d  ⇒  dL/dδ = −g / (√c·√(δ(δ+2)))
        w = -grad / (np.sqrt(c) * np.sqrt(np.maximum(delta * (delta + 2.0), _EPS)))

        # δ = 2cS/(αβ) with S = ‖z−p‖², α/β functions of z/p norms:
        # ∂δ/∂z = 4c(z−p)/(αβ) + 2cδz/α  (and symmetrically for p)
        A = 4.0 * c * w / (alpha * beta)  # (N, K)
        wd = w * delta
        dz = A.sum(axis=1, keepdims=True) * z - A @ p \
            + (2.0 * c / alpha) * wd.sum(axis=1, keepdims=True) * z
        dp = A.sum(axis=0)[:, None] * p - A.T @ z \
            + (2.0 * c / beta.T) * wd.sum(axis=0)[:, None] * p

        self.P.grad += _expmap0_backward(dp, self.P.data, self._r_p, self._f_p, c)
        return _expmap0_backward(dz, self._x, self._r_x, self._f_x, c)
