"""
Checkpoint utilities: snapshot, restore, and average model parameters.

A "state" is a plain list of arrays in `model.parameters()` order — the same
ordering contract the optimizers rely on. Averaging states (over an annealing
window, or SWA-style over late training) often beats the last noisy checkpoint.

`get_state`/`set_state`/`average_states` work in memory; `save`/`load` persist
to a `.npz` on disk so a long run survives the process that produced it.
"""
import numpy as _np  # real numpy: file IO is always plain arrays

from core.backend import xp as np, to_numpy


def get_state(model) -> list:
    """Snapshot all parameters as copies (safe to keep while training continues)."""
    return [np.array(p.data) for p in model.parameters()]


def set_state(model, state: list) -> None:
    """Restore a snapshot taken by get_state on the same architecture."""
    for p, arr in zip(model.parameters(), state, strict=True):
        p.data = arr


def average_states(states: list[list]) -> list:
    """Elementwise mean of several states (checkpoint averaging)."""
    n = len(states)
    return [sum(arrs) / n for arrs in zip(*states, strict=True)]


def save(path, model, **metadata) -> str:
    """Write a model's parameters to a `.npz` file, plus optional metadata.

    Parameters are stored under positional keys (`p0`, `p1`, …) because a
    "state" is defined by `model.parameters()` *order*, not by name. Extra
    keyword arguments (step, loss, lr, …) are stored alongside so a run can
    report where a checkpoint came from:

        save("ckpt.npz", model, step=1500, loss=2.31)

    Always writes real numpy arrays, so a checkpoint saved under the jax
    backend loads under numpy and vice versa.
    """
    path = str(path)
    if not path.endswith(".npz"):
        path += ".npz"
    arrays = {f"p{i}": _np.asarray(to_numpy(p.data))
              for i, p in enumerate(model.parameters())}
    meta = {f"meta_{k}": _np.asarray(v) for k, v in metadata.items()}
    _np.savez(path, **arrays, **meta)
    return path


def load(path, model) -> dict:
    """Restore parameters saved by `save` into an identical architecture.

    Returns the metadata dict that was passed to `save`. Raises if the
    checkpoint does not match the model's parameter count or shapes — silently
    loading a mismatched checkpoint is the kind of bug that costs a day.
    """
    with _np.load(str(path)) as data:
        params = model.parameters()
        saved = [k for k in data.files if k.startswith("p")]
        if len(saved) != len(params):
            raise ValueError(
                f"checkpoint has {len(saved)} parameters, model has {len(params)}"
            )
        for i, p in enumerate(params):
            arr = data[f"p{i}"]
            expected = tuple(_np.shape(to_numpy(p.data)))
            if arr.shape != expected:
                raise ValueError(
                    f"parameter {i} shape mismatch: checkpoint {arr.shape} "
                    f"vs model {expected}"
                )
            p.data = np.asarray(arr)
        return {
            k[len("meta_"):]: data[k].item() if data[k].ndim == 0 else data[k]
            for k in data.files
            if k.startswith("meta_")
        }
