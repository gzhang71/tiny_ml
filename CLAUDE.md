# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running examples

All commands are run from the repo root (`tiny_ml/`) with the venv Python:

```bash
# Run any example
.venv/bin/python -c "import sys; sys.path.insert(0, '..'); from examples.mlp import main; main()"

# Or from the parent directory
cd .. && tiny_ml/.venv/bin/python -m tiny_ml.examples.mlp
```

Because the package root is `tiny_ml/`, Python must be invoked from the **parent directory** for `-m` module syntax to work, or the parent must be added to `sys.path` manually.

## Architecture

This is a pure-numpy autodiff library. Every object participates in a manual forward/backward computation graph — there is no autograd engine.

### Base classes (`core/`)

- `Parameter` — wraps a numpy array with a `.grad` field; the only leaf that holds weights
- `Module` — base for everything; `parameters()` recurses into `__dict__` collecting all `Parameter` and nested `Module` instances automatically
- `Layer(Module)` — building block (no semantics beyond naming)
- `Model(Module)` — top-level model (no semantics beyond naming)
- `Loss` — separate hierarchy; `forward` returns a scalar, `backward` returns `d_loss/d_pred` with no arguments (stores state from the last forward)
- `Optimizer` — takes a flat list of `Parameter` objects and mutates `.data` in `step()`

### Import convention

All internal imports use **package-relative paths without the `tiny_ml.` prefix**:
```python
from core.module import Layer
from layers.linear import Linear
```
This requires the repo root (`tiny_ml/`) to be on `sys.path`.

### Array backend (numpy / JAX)

Library modules never import numpy directly — they use `from core.backend import xp as np`,
which resolves to numpy (default) or `jax.numpy` when `TINY_ML_BACKEND=jax` is set before
import. `TINY_ML_JAX_X64=0` switches JAX to float32 (faster; float64 default matches numpy
bit-for-bit). Rules that keep code backend-portable:

- **No in-place array mutation.** JAX arrays are immutable. Attribute-level `p.grad += g` is
  fine (Python rebinds), but `arr[idx] = v` / `arr[idx] += v` / `np.add.at` are not — use
  `backend.scatter_add(arr, idx, updates)` and assign the result back.
- **No `np.random`.** `jax.numpy` has no `random` module. Use `backend.randn(*shape)` for
  init/noise and `backend.sample_categorical(probs)` for sampling — both draw from real
  numpy's RNG so `np.random.seed()` behaves identically in both modes.
- `examples/`, `training/trainer.py`, and `metrics/` keep plain numpy (bookkeeping only).
- **Keep shapes static in cached/decode paths.** JAX caches compiled ops per shape *and* per
  baked-in constant, so anything that changes each decode step (cache length, position
  offset) must go through `backend.write_slice`/`take_slice` (runtime indices), never
  Python-slice indexing or growing `concatenate`.
- JAX mode speeds up **training and batched forward passes** (~2x at float32). `generate()`
  works at full speed thanks to the static KV cache, though tiny models still decode faster
  on numpy (eager JAX pays per-op dispatch on hundreds of small ops per token).

### Layer contract

Each layer stores the inputs it needs for backward in `self._<name>` during `forward`, then uses them in `backward`. There is no tape — if you call `forward` twice before `backward`, the second call overwrites the saved state.

### Key design choices

- **`_TiedProjection`** in `layers/attention.py` — shares the `Embedding.W` Parameter between the embedding lookup and the output projection (weight tying). It owns no parameters of its own; `parameters()` returns `[]` to avoid double-counting.
- **`T5.backward`** bypasses `Embedding.backward` for the shared embedding and calls `np.add.at` directly, because the same embedding is called twice (encoder and decoder) and `_tokens` would be overwritten.
- **`TransformerBlock` backward order** — the residual connection means grad flows through both the sublayer and the identity path: `grad = grad + norm.backward(sublayer.backward(grad))`.
- **Softmax** is implemented as module-level functions (`_softmax`, `_softmax_backward`) in `layers/attention.py` rather than a `Layer` because it is always inlined inside attention with a causal mask and scale that attention owns.
- **`VAE`** uses `MLP` for encoder/decoder (not a private `_build_mlp` helper).
- **KV cache (inference-only, static)** — attention layers take `forward(x, use_cache=True)`: self-attention writes new K/V into a cache **preallocated at `max_cache_len`** (`backend.write_slice`) and masks by absolute position, which hides both future tokens and unwritten padded slots (padding contributes exactly 0 after softmax, so results are exact). Static shapes matter: in jax mode a growing cache would recompile every op on every decode step. The jax path attends over the full padded cache; the numpy path slices to the valid prefix since it has no compile cache to protect. `CrossAttention` computes encoder K/V once and reuses them. Positional embeddings take an `offset` (applied via `backend.take_slice` — same recompile concern); models track it in `_cache_len`. `generate()` prefills on the prompt then decodes one token per step, collecting tokens in a Python list (a growing array concat would also recompile per step). `reset_cache()` clears everything. `backward` assumes the last forward was uncached — never train with `use_cache=True`.

### Directory layout

```
core/        — Parameter, Module, Layer, Model, Loss, Optimizer base classes
layers/      — reusable building blocks (Linear, activations, LayerNorm,
               Embedding variants, FeedForward, ResidualBlock, attention classes)
models/      — full models composed from layers (MLP, ResNet, Sequential,
               Transformer, GPT2, T5, VAE)
losses/      — MSELoss, SoftmaxCrossEntropy, BinaryCrossEntropy
optim/       — SGD, Momentum, ADAM
metrics/     — precision, recall, f1_score, accuracy (binary classification)
training/    — Trainer (fit / predict / evaluate loop)
examples/    — one runnable script per model
```
