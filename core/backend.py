"""Array backend selection: numpy (default) or JAX.

Select the backend with an environment variable, set BEFORE importing any
tiny-pre-train module:

    TINY_PRE_TRAIN_BACKEND=jax python -c "from examples.gpt2 import main; main()"

In jax mode every `np.` call inside the library resolves to `jax.numpy`, so
matmuls and elementwise ops run through XLA. float64 is enabled by default so
results match numpy exactly; set TINY_PRE_TRAIN_JAX_X64=0 for float32, which is
considerably faster and the usual choice when speed is the point.

JAX arrays are immutable, so the library never mutates arrays in place:
gradient accumulation rebinds attributes (`p.grad = p.grad + g`, which is what
`p.grad += g` falls back to), and scattered writes go through `scatter_add`
below. Randomness always comes from real numpy (`randn`, `sample_categorical`)
so seeding behaves identically in both modes.
"""
import os

import numpy as _np

BACKEND = os.environ.get("TINY_PRE_TRAIN_BACKEND", "numpy").lower()

if BACKEND == "jax":
    from jax import config as _jax_config
    _jax_config.update("jax_enable_x64", os.environ.get("TINY_PRE_TRAIN_JAX_X64", "1") != "0")
    import jax.numpy as xp
    from jax import lax as _lax
else:
    BACKEND = "numpy"
    xp = _np


def scatter_add(arr, index, updates):
    """`arr[index] += updates` that works on both backends.

    Returns the updated array — always assign the result back
    (`a = scatter_add(a, idx, u)`); numpy mutates in place, JAX cannot.
    `index` may be anything numpy fancy indexing accepts, including slices
    and tuples of arrays.
    """
    if BACKEND == "jax":
        # asarray: tolerate plain-numpy targets (e.g. user-zeroed grads)
        return xp.asarray(arr).at[index].add(updates)
    _np.add.at(arr, index, updates)
    return arr


def take_slice(arr, start, size, axis=0):
    """`arr[..., start:start+size, ...]` along `axis`.

    In jax mode this lowers to `lax.dynamic_slice` with a *runtime* start
    index; a plain `arr[start:...]` bakes the offset into the compiled op, so
    a start that changes every decode step would trigger a recompile per step.
    """
    if BACKEND == "jax":
        return _lax.dynamic_slice_in_dim(arr, start, size, axis)
    idx = [slice(None)] * arr.ndim
    idx[axis] = slice(start, start + size)
    return arr[tuple(idx)]


def write_slice(arr, update, start, axis=0):
    """`arr[..., start:start+len, ...] = update` along `axis`, returning the
    result — always assign it back. Runtime start index for the same reason
    as `take_slice`; numpy mutates in place.
    """
    if BACKEND == "jax":
        return _lax.dynamic_update_slice_in_dim(arr, update, start, axis)
    idx = [slice(None)] * arr.ndim
    idx[axis] = slice(start, start + update.shape[axis])
    arr[tuple(idx)] = update
    return arr


def randn(*shape):
    """Standard-normal sample as a backend array, drawn with numpy's RNG
    so `np.random.seed(...)` gives identical draws in both modes."""
    return xp.asarray(_np.random.randn(*shape))


def sample_categorical(probs) -> int:
    """Draw one index from a probability vector (any backend's array)."""
    p = _np.asarray(probs, dtype=_np.float64)
    return int(_np.random.choice(p.size, p=p / p.sum()))


def to_numpy(x) -> _np.ndarray:
    return _np.asarray(x)
