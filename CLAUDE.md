# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running examples and tests

All commands are run from the repo root (`tiny-pre-train/`) with the venv Python:

```bash
.venv/bin/python -m examples.mlp     # run any example
.venv/bin/python -m tests.run_all    # run the whole test suite
```

The test suite has **no third-party dependencies** (there is no pytest in the venv) —
`tests/runner.py` collects `test_*` functions itself. Keep it that way; test functions
are named so that `pytest tests/` also works for anyone who has it.

**When you change or add a layer, add a gradient check.** `tests/gradcheck.py` compares
each hand-derived backward against central differences; `check_layer(layer, x)` covers
both the input gradient and every `Parameter.grad`. It requires float64, so run it under
numpy or `TINY_PRE_TRAIN_JAX_X64=1`. Two gotchas worth knowing:

- The comparison has an **absolute floor** (`_ATOL`) as well as a relative tolerance.
  Finite differences of a genuinely-zero gradient return ~1e-9 of roundoff, which a pure
  relative metric reports as 100% error.
- A layer whose `backward` deliberately injects an extra term (`MoEFeedForward`'s aux
  loss) will not match a check of the main objective alone — the objective must include
  that term too. See `test_moe_aux_loss_gradient`.

The repo directory name (`tiny-pre-train`) is **not** a valid Python identifier, so the package
cannot be imported as a whole (`python -m tiny-pre-train.examples.mlp` is a syntax error). Run
from the repo root instead: the top-level directories (`core/`, `layers/`, …) are the
importable packages, and the cwd on `sys.path` is what makes `from core.module import Layer`
resolve. `pip install -e .` (see `pyproject.toml`) installs those directories as
top-level packages, which lifts the run-from-repo-root requirement.

## Architecture

This is a pure-numpy autodiff library. Every object participates in a manual forward/backward computation graph — there is no autograd engine.

### Base classes (`core/`)

- `Parameter` — wraps a numpy array with a `.grad` field; the only leaf that holds weights
- `Module` — base for everything; `parameters()` recurses into `__dict__` collecting all `Parameter` and nested `Module` instances automatically
- `Layer(Module)` — building block (no semantics beyond naming)
- `Model(Module)` — top-level model (no semantics beyond naming)
- `Loss` — separate hierarchy; `forward` returns a scalar, `backward` returns `d_loss/d_pred` with no arguments (stores state from the last forward)
- `Optimizer` — takes a flat list of `Parameter` objects and mutates `.data` in `step()`;
  `parameters()` exposes that list (used by gradient clipping)
- **Train/eval mode** — `Module.training` (class-level default `True`), set recursively by
  `model.train()` / `model.eval()`. `Module.modules()` does the traversal by walking
  `__dict__` the same way `parameters()` does — it deliberately does *not* go through
  `parameters()`, since layers like `Attention` override that to return a hand-built list
  and `_TiedProjection` returns nothing at all. Only `Dropout` reads the flag today

### Import convention

All internal imports use **paths relative to the repo root, with no top-level package prefix**:
```python
from core.module import Layer
from layers.linear import Linear
```
This requires the repo root (`tiny-pre-train/`) to be on `sys.path`.

### Array backend (numpy / JAX)

Library modules never import numpy directly — they use `from core.backend import xp as np`,
which resolves to numpy (default) or `jax.numpy` when `TINY_PRE_TRAIN_BACKEND=jax` is set before
import. `TINY_PRE_TRAIN_JAX_X64=0` switches JAX to float32 (faster; float64 default matches numpy
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
- **`Attention` base class** in `layers/attention.py` — owns the Q/K/V/O projections, head split/merge, the static KV cache, and the dense softmax forward/backward. Variants derive by overriding hooks, not `forward`/`backward`: `_attend`/`_attend_backward` replace the attention core between the projections (FlashAttention), `_score_bias`/`_score_bias_backward` add an additive score bias in both uncached and cached paths (T5SelfAttention), `_position_encode`/`_position_encode_backward` transform per-head Q/K right after the projections (RoPEAttention rotates by absolute position; when decoding, `offset` is the cache length, so cached keys are stored already rotated). `RotaryPositionalEmbedding` in `layers/embedding.py` holds the rotation math (interleaved pairs, RoFormer convention); unlike the additive positional embeddings it is applied inside attention, so RoPE models need no input-level PE. The hooks are orthogonal — a RoPE + flash variant is just a mixin of `RoPEAttention` and `FlashAttention2`. `MultiHeadAttention` is the base behavior under its standard name; `CrossAttention` reuses the core but overrides `forward`/`backward` for its two-input signature and tuple gradient. `TransformerBlock` takes `attention_cls=` to swap implementations.
- **`FlashAttention` / `FlashAttention2`** in `layers/flash_attention.py` — `Attention` subclasses (overriding only `_attend`/`_attend_backward`) computing exact attention with the tiled online-softmax algorithm (v1: KV-outer loop with per-tile output renormalization; v2: Q-outer loop, unnormalized accumulator, one divide per Q block). Forward saves only O and the logsumexp L; backward recomputes tile probabilities via `P = exp(S − L)` and `D = rowsum(dO ∘ O)`, so per-head extra memory is O(T) not O(T²). No speed win in numpy — the algorithm is the point. `use_cache=True` falls back to the inherited standard cached path.
- **`Head` family** in `layers/heads.py` — classification heads mapping features `(..., in_features)` to logits `(..., n_classes)`. The base class owns the interface and leading-dim flattening; subclasses override the `_logits`/`_logits_backward` hooks on 2-D arrays (same pattern as `Attention`). Variants: `LinearHead` (wraps `Linear`), `EuclideanHead` (negative *squared* distance to learnable prototypes — squared to keep the gradient finite at zero distance), `CosineHead` (L2-normalized features × L2-normalized class weights, learnable 0-d scale Parameter; label-dependent margins like ArcFace belong in the loss, not the layer), and `HyperbolicHead` (Poincaré ball: exp₀ then negative geodesic distance to prototypes). Hyperbolic prototypes are stored as *tangent-space* vectors and mapped through exp₀ each forward, so the Euclidean optimizers in `optim/` work unchanged (no Riemannian SGD) and parameters can never leave the ball — tanh in exp₀ makes projection/clipping unnecessary. `_expmap0_backward` switches to the analytic r→0 limit below `√c·r = 1e-3` because the closed form loses all precision to cancellation there.
- **`MoEFeedForward`** in `layers/moe.py` — top-k routed mixture-of-experts FFN,
  drop-in for `FeedForward` (swap post-hoc via `block.ffn = ...`). *Dense* educational
  implementation: every expert runs on every token and outputs are mixed by renormalized
  top-k softmax gates — no scatter/dispatch, so it is backend-safe and shape-static (the
  routing math is the point, not the FLOP savings, same spirit as FlashAttention). The
  top-k mask is treated as constant in backward (correct a.e.). `from_dense()` upcycles a
  trained FFN: experts are deep copies, router is zero → output exactly equals the dense
  FFN, but that configuration is a stationary point (identical experts ⇒ identical expert
  grads and exactly-zero router grad), so pass `router_scale>0` or jitter the router
  in place to let experts specialize. `aux_coef>0` enables the Switch-style
  load-balancing loss α·E·Σ f_e·P_e (f = hard routing fractions, constant; P = mean
  router probs): `forward` stores the scalar in `.aux_loss` for reporting, and
  `backward` injects its gradient into the router directly, so the upstream grad
  stays that of the main loss. Without it top-k routing collapses onto 1-2 experts.
- **`Dropout`** in `layers/dropout.py` — inverted dropout: the mask is applied *and*
  rescaled by 1/(1−p) at training time, so `forward` is an exact identity once
  `model.eval()` is called. It is the only layer whose behavior depends on
  `Module.training`. The mask is drawn via `backend.rand` (numpy's RNG in both
  backends, so seeding is reproducible) and saved for backward, per the layer contract.
- **Grouped-query attention** — `Attention(..., n_kv_heads=k)` gives each group of
  `n_heads // k` query heads one shared K/V head (`k=1` is multi-query attention). The
  K/V projections shrink to `k · d_k` and **the KV cache holds only `k` heads**, which is
  the point: KV cache size dominates long-context decoding. Expansion happens in
  `_repeat_kv` *between* the projections and the attention core, so every `_attend`
  implementation — including the tiled FlashAttention kernels — is oblivious to GQA.
  `_repeat_kv_backward` sums each group's gradients back onto its shared head.
- **Key padding masks** — `forward(x, key_padding_mask=mask)` where `mask` is `(B, T_k)`
  and **True marks padding to ignore** (PyTorch's convention). Reshaped to `(B, 1, 1, T_k)`
  so it broadcasts over heads and queries; applied in the dense path, the cached decode
  path, and the flash kernels (per-tile key slice in `_masked_scores`). Masked entries get
  −1e9, which underflows to exactly 0 after the exp, so backward needs no separate
  masking. `TransformerBlock` and `GPT2` plumb the argument through; `Transformer` and
  `T5` do not yet.
- **KV cache (inference-only, static)** — attention layers take `forward(x, use_cache=True)`: self-attention writes new K/V into a cache **preallocated at `max_cache_len`** (`backend.write_slice`) and masks by absolute position, which hides both future tokens and unwritten padded slots (padding contributes exactly 0 after softmax, so results are exact). Static shapes matter: in jax mode a growing cache would recompile every op on every decode step. The jax path attends over the full padded cache; the numpy path slices to the valid prefix since it has no compile cache to protect. `CrossAttention` computes encoder K/V once and reuses them. Positional embeddings take an `offset` (applied via `backend.take_slice` — same recompile concern); models track it in `_cache_len`. `generate()` prefills on the prompt then decodes one token per step, collecting tokens in a Python list (a growing array concat would also recompile per step). `reset_cache()` clears everything. `backward` assumes the last forward was uncached — never train with `use_cache=True`.

### Directory layout

```
core/        — Parameter, Module, Layer, Model, Loss, Optimizer base classes
layers/      — reusable building blocks (Linear, activations, LayerNorm, RMSNorm,
               Embedding variants, FeedForward, SwiGLU, MoEFeedForward, Dropout,
               ResidualBlock, attention classes, Head variants)
models/      — full models composed from layers (MLP, ResNet, Sequential,
               Transformer, GPT2, T5, VAE)
losses/      — MSELoss, SoftmaxCrossEntropy, BinaryCrossEntropy, BCEWithLogits.
               SoftmaxCrossEntropy picks the target form by *shape* (labels are
               the logits shape minus the class axis; one-hot is the full logits
               shape) and raises otherwise — sniffing `ndim` instead silently
               misreads (B, T) labels as one-hot. It averages over all leading
               dims, so (B, T, C) and its flattened (B*T, C) form agree.
optim/       — SGD, Momentum, ADAM, AdamW (+ decay_groups), schedule.py
               (ConstantLR, LinearWarmup, CosineWithWarmup, InverseSqrt — a
               schedule is just a callable step->lr), clip.py (clip_grad_norm)
metrics/     — precision, recall, f1_score, accuracy (binary classification)
training/    — Trainer (fit / predict / evaluate; optional lr_schedule,
               max_grad_norm, grad_accum_steps, val_data). Accumulation scales
               each micro-batch's loss gradient by 1/k so k micro-batches equal
               one k-times-larger batch; `Trainer.step` counts optimizer steps,
               not micro-batches, so schedules stay correct. checkpoint.py
               (get_state/set_state/average_states — a "state" is a list of
               array copies in model.parameters() order — plus save/load to .npz,
               which always writes real numpy so checkpoints cross backends)
tests/       — gradcheck.py (finite differences) + test_gradients.py,
               test_invariants.py, test_training.py; runner.py is the
               dependency-free collector, `python -m tests.run_all` runs everything
examples/    — one runnable script per model
```
