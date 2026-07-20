# tiny-pre-train

A from-scratch machine learning library built on pure NumPy — no PyTorch, no TensorFlow. Implements the full stack from basic autodiff primitives up to GPT-2 and T5.

## What's implemented

**Layers**
- `Linear` — fully connected layer with He initialization
- Activations: `ReLU`, `Sigmoid`, `Tanh`, `SiLU`, `GeLU`
- `LayerNorm` — with learnable `gamma`/`beta`
- `RMSNorm` — LayerNorm without mean-centering or bias (Llama-style)
- `Embedding`, `SinusoidalPositionalEmbedding`, `LearnedPositionalEmbedding`, `FeatureEmbedding`
- `FeedForward` — position-wise FFN with configurable activation and expansion factor
- `SwiGLU` — gated FFN `W2(SiLU(W1x) ⊙ W3x)`; swap into `TransformerBlock` via `ffn_cls=SwiGLU`
- `MoEFeedForward` — mixture-of-experts FFN with top-k routing; `from_dense()` upcycles a trained dense FFN into experts
- `ResidualBlock` — two-layer MLP-style residual block
- `Dropout` — inverted dropout; identity in eval mode (`model.eval()`)
- `MultiHeadAttention` — scaled dot-product with optional causal mask, grouped-query
  attention (`n_kv_heads=`), and key padding masks
- `TransformerBlock` — pre-norm residual (attention + FFN)
- `T5SelfAttention`, `CrossAttention`, `RelativePositionBias` — T5-specific attention variants

**Models**
- `MLP` — arbitrary-depth multilayer perceptron
- `Sequential` — ordered layer container
- `ResNet` — stack of residual blocks with input/output projections
- `Transformer` — GPT-style decoder-only transformer (sinusoidal positional encoding)
- `GPT2` — decoder-only with learned positional embeddings, GeLU FFN, weight-tied output projection, and autoregressive `generate()`
- `T5` — encoder-decoder with relative position bias, shared embeddings, and weight-tied output head
- `VAE` — variational autoencoder with reparameterization trick and KL loss

**Training infrastructure**
- Losses: `MSELoss`, `SoftmaxCrossEntropy`, `BinaryCrossEntropy`, `BCEWithLogits`
- Optimizers: `SGD`, `Momentum`, `ADAM`, `AdamW` (decoupled weight decay)
- LR schedules: `ConstantLR`, `LinearWarmup`, `CosineWithWarmup`, `InverseSqrt`
- `clip_grad_norm` — global-norm gradient clipping
- Metrics: `precision`, `recall`, `f1_score`, `accuracy`
- `Trainer` — batched fit/predict/evaluate loop, with optional LR schedule,
  gradient clipping, gradient accumulation, and held-out validation loss
- Checkpoint utils — `get_state` / `set_state` / `average_states` (checkpoint
  averaging), plus `save` / `load` to `.npz` on disk

**Tests**
- `tests/gradcheck.py` — finite-difference gradient checking; every layer, head,
  and loss is verified against central differences
- equivalence tests: FlashAttention ≡ dense attention, cached decoding ≡ a full
  forward pass, jax backend ≡ numpy backend

## Running examples

Each model has a self-contained example in `examples/`. Run from the repo root (`tiny-pre-train/`):

```bash
python -m examples.mlp          # spiral classification
python -m examples.sequential   # sine wave regression
python -m examples.resnet       # checkerboard classification
python -m examples.transformer  # next-token prediction
python -m examples.gpt2         # token generation
python -m examples.t5           # seq2seq copy task
python -m examples.vae          # 2D cluster reconstruction
python -m examples.checkpoint_averaging  # averaged snapshots beat the last one
python -m examples.moe_upcycle  # dense→MoE upcycling, then expert specialization
python -m examples.train_100m       # 113.8M-param dense GPT-2 on this repo's source
python -m examples.train_100m_moe   # 170.5M-param MoE GPT-2 (94.9M active/token)
```

## Workflow

The typical path from data to trained model is four steps: build a model from the pieces in `models/` (or compose your own from `layers/`), pick a loss and an optimizer, hand all three to `Trainer`, then evaluate or predict. From `examples/mlp.py`:

```python
import numpy as np
from models.mlp import MLP
from losses.losses import SoftmaxCrossEntropy
from optim.adam import ADAM
from training.trainer import Trainer

# 1. model — layer sizes: 2 inputs → two hidden layers → 2 classes
model = MLP([2, 64, 32, 2])

# 2. loss + optimizer — the optimizer takes the flat parameter list
trainer = Trainer(model, SoftmaxCrossEntropy(), ADAM(model.parameters(), lr=3e-3))

# 3. train — shuffles and batches internally
trainer.fit(x, y, epochs=100, batch_size=64)

# 4. evaluate / predict
logits = trainer.predict(x)
accuracy = (logits.argmax(axis=1) == y).mean()
```

If you need more control than `fit()` gives you (gradient inspection, multi-input models
like T5), drop down to `trainer.train_step(x, y)` per batch, or write the five-line loop
yourself — see the next section. For the generative models, skip `Trainer` for inference
and call `model.generate(...)` directly (see `examples/gpt2.py` and `examples/t5.py`).

For a real pre-training run, `Trainer` takes the usual machinery directly:

```python
from optim.adamw import AdamW, decay_groups
from optim.schedule import CosineWithWarmup

_, no_decay = decay_groups(model)          # skip biases and LayerNorm gains
optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, no_decay=no_decay)

trainer = Trainer(
    model, SoftmaxCrossEntropy(), optimizer,
    lr_schedule=CosineWithWarmup(peak_lr=3e-4, warmup_steps=100, total_steps=5000),
    max_grad_norm=1.0,        # global-norm clipping; trainer.last_grad_norm to log it
    grad_accum_steps=4,       # effective batch = batch_size x 4
)
history = trainer.fit(x, y, epochs=10, batch_size=32, val_data=(x_val, y_val))
```

`examples/train_100m.py` uses exactly this setup, plus periodic `checkpoint.save()`.

## Testing

Every backward pass in this library is derived by hand, so the test suite's main job is
checking each one against finite differences:

```bash
python -m tests.run_all          # no dependencies needed
pytest tests/                    # also works, if you have pytest
```

`tests/gradcheck.py` perturbs each parameter entry by +-eps and compares the resulting
central difference against the analytic gradient, for every layer, head, and loss. The
rest of the suite pins the equivalences the design depends on: FlashAttention against
dense attention, incremental KV-cached decoding against a full forward pass, GQA against
tiled multi-head attention, padding masks against the equivalent shorter sequence, and
the jax backend against numpy. Run it under either backend:

```bash
TINY_PRE_TRAIN_BACKEND=jax python -m tests.run_all
```

## How it works

There is no autograd engine. Every layer manually implements `forward` and `backward`, saving the tensors it needs for the gradient computation during the forward pass. The backward pass is called explicitly after computing the loss gradient.

```python
# Training loop (what Trainer does internally)
optimizer.zero_grad()
pred = model.forward(x)
loss = loss_fn.forward(pred, y)
grad = loss_fn.backward()       # d_loss / d_pred
model.backward(grad)            # populates .grad on all Parameters
optimizer.step()                # mutates .data on all Parameters
```

`Module.parameters()` collects `Parameter` objects by recursing through `__dict__`, so composite modules automatically expose all their leaf parameters without any registration boilerplate.

## Architecture

```
tiny-pre-train/
├── core/                  # foundations everything else builds on
│   ├── parameter.py       #   Parameter — numpy array + .grad field (the only weight leaf)
│   ├── module.py          #   Module, Layer, Model, Loss, Optimizer base classes
│   └── backend.py         #   array backend (numpy or jax.numpy via TINY_PRE_TRAIN_BACKEND)
├── layers/                # reusable building blocks
│   ├── linear.py          #   Linear
│   ├── activations.py     #   ReLU, Sigmoid, Tanh, SiLU, GeLU
│   ├── normalization.py   #   LayerNorm, RMSNorm
│   ├── embedding.py       #   Embedding + positional/feature variants
│   ├── feedforward.py     #   FeedForward, SwiGLU (position-wise FFNs)
│   ├── moe.py             #   MoEFeedForward (top-k routed mixture of experts)
│   ├── residual.py        #   ResidualBlock
│   ├── dropout.py         #   Dropout (train/eval aware)
│   └── attention.py       #   MultiHeadAttention (+GQA, padding masks), TransformerBlock
├── models/                # full models composed from layers
│   ├── mlp.py             #   MLP
│   ├── sequential.py      #   Sequential
│   ├── resnet.py          #   ResNet
│   ├── transformer.py     #   Transformer (GPT-style decoder-only)
│   ├── gpt2.py            #   GPT2 (learned positions, weight tying, generate())
│   ├── t5.py              #   T5 (encoder-decoder)
│   └── vae.py             #   VAE
├── losses/                # MSELoss, SoftmaxCrossEntropy, BinaryCrossEntropy, BCEWithLogits
├── optim/                 # SGD, Momentum, ADAM, AdamW + schedules + gradient clipping
├── metrics/               # precision, recall, f1_score, accuracy
├── training/              # Trainer (fit / predict / evaluate loop), checkpoint utils
├── tests/                 # gradient checks + equivalence tests (no dependencies)
└── examples/              # one runnable script per model
```

**Class hierarchy.** `Module` is the root: its `parameters()` recurses through `__dict__`, collecting every `Parameter` and nested `Module`, so composition alone wires up the parameter tree. `Layer` and `Model` subclass `Module` purely for naming — layers are building blocks, models are top-level compositions. `Loss` is a separate hierarchy (`forward` returns a scalar, `backward` returns `d_loss/d_pred` from stored state), and `Optimizer` takes the flat parameter list and mutates `.data` in `step()`. `Module.train()` / `Module.eval()` set `self.training` recursively over the same tree, which is what makes `Dropout` an identity at inference.

**Data flow.** Each layer stores whatever `forward` computed that `backward` needs in `self._<name>` attributes — there is no tape, so calling `forward` twice before `backward` overwrites that state. Gradients flow top-down: the loss produces the initial gradient, each module's `backward` populates `.grad` on its own parameters and returns the gradient for its input.

**Backend isolation.** Library code never imports numpy directly; everything goes through `core/backend.py` (`from core.backend import xp as np`), which is what lets the same code run on numpy or JAX. Examples, `Trainer`, and metrics use plain numpy since they only do bookkeeping.

## JAX backend (optional speedup)

The library runs on numpy by default. Set `TINY_PRE_TRAIN_BACKEND=jax` to route every array
operation through `jax.numpy`/XLA instead — no code changes needed:

```bash
TINY_PRE_TRAIN_BACKEND=jax python -m examples.gpt2      # float64, matches numpy exactly
TINY_PRE_TRAIN_BACKEND=jax TINY_PRE_TRAIN_JAX_X64=0 python -m examples.gpt2  # float32, fastest
```

With the same seed, jax float64 mode reproduces numpy results (bit-for-bit for GPT-2 /
Transformer; T5 drifts in the last bits because scatter-add accumulation order differs).
Rough numbers on an Apple-silicon CPU for a GPT-2-style model (d_model=512, 6 layers,
batch 8, seq 256):

| backend        | train step |
|----------------|-----------:|
| numpy          |    1.26 s  |
| jax (float64)  |    1.29 s  |
| jax (float32)  |    0.65 s  |

Use JAX mode for **training and batched forward passes**, and bigger wins are expected on
GPU/TPU. `generate()` uses a **static KV cache** (preallocated at `max_seq_len`) so array
shapes never change between decode steps — without it, XLA would recompile every op for
every new sequence length, which made generation ~50x slower. Small models still decode
somewhat faster on numpy, since eager JAX pays per-op dispatch overhead on every step.

## Benchmarks: 100M-scale training

Two byte-level language models (vocab = 256) trained on this repo's own source code
(~230 KB of Python + Markdown — the corpus is the repo, so it grows as the repo does)
with `examples/train_100m.py` and `examples/train_100m_moe.py`. Setup: seq len 128,
batch 8, AdamW lr 3e-4 with weight decay 0.1, cosine schedule with 7-step warmup
decaying to 3e-5, gradient clipping at global norm 1.0, 150 steps, JAX float32
backend, Apple M3 Max (CPU). The last 5% of the corpus is held out for validation.
The MoE model swaps each block's dense FFN for a 4-expert top-2 `MoEFeedForward` and
halves the layer count — more *total* parameters than the dense model, fewer *active*
per token. (Note the MoE implementation is dense-compute, so step time does not
benefit from the sparsity; see the design note in `layers/moe.py`.)

| model | config | params (total) | params (active/token) | step time | train loss 1 → 150 | val loss | val ppl |
|---|---|--:|--:|--:|--|--:|--:|
| dense GPT-2 | d768, 12h, 16L | 113,800,704 | 113,800,704 | 1.36 s | 5.84 → 2.87 | 3.33 | 27.9 |
| MoE GPT-2 | d768, 12h, 8L, 4e top-2 | 170,460,704 | 94,901,792 | 1.58 s | 5.75 → 2.87 | 3.27 | 26.4 |

Single-step train losses are noisy at batch 8 (nearby steps range roughly 2.4–3.1);
averaged over the last five logged steps they are 2.82 dense and 2.71 MoE. The
held-out numbers are the ones to compare — they average 5 fixed validation batches,
and the ~0.45 nat gap between train and validation loss is the overfitting you would
expect after 150 steps on a 230 KB corpus.

150 steps ≈ 150K tokens seen — enough for loss to fall well below the uniform-random
5.55 and for samples to pick up code-shaped structure (indentation, `self`, call
syntax), not enough for real code. Both scripts accept `TRAIN_STEPS=` to go longer.
The MoE run trains with the Switch-style load-balancing aux loss
(`MoEFeedForward(aux_coef=0.01)`, the script's default), which keeps routing spread
across experts (per-block gate mass typically spread like 0.30/0.29/0.26/0.14). Set `AUX_COEF=0`
to watch the routers collapse onto one or two dominant experts
(0.98/0.01/0.00/0.00) — the rich-get-richer failure mode aux losses exist to
prevent. Honest caveat: at this tiny scale the collapsed run has previously reached a
*lower* train loss than the balanced one — the balance tax is real, and its payoff
(all experts trained, even device load under sparse dispatch) only shows up at
scale. This trade-off is why DeepSeek-V3 moved to aux-loss-free balancing.

### Parameter breakdown

Both models share the embedding/attention skeleton; they differ only in FFN and depth.
The output head is weight-tied to the token embedding, so it contributes no parameters.

| component | dense (16 layers) | MoE (8 layers) |
|---|--:|--:|
| token embedding (tied output head) | 196,608 | 196,608 |
| learned positional embedding | 196,608 | 196,608 |
| attention Q/K/V/O — per block | 2,362,368 | 2,362,368 |
| FFN — per block | 4,722,432 | 18,892,804 = router 3,076 + 4 × 4,722,432 |
| 2 LayerNorms — per block | 3,072 | 3,072 |
| block total × depth | 7,087,872 × 16 | 21,258,244 × 8 |
| final LayerNorm | 1,536 | 1,536 |
| **total** | **113,800,704** | **170,460,704** |

### Training-time array accounting

Every `Parameter` carries its weights (used in forward) and a same-shaped `.grad`
(written in backward); ADAM keeps two moment arrays per parameter. Training memory is
therefore 4× the model size, plus activations; inference needs only the weights.

| array set | role | count | dense | MoE |
|---|---|--:|--:|--:|
| `p.data` | forward (weights) | 1× params | 113.8M | 170.5M |
| `p.grad` | backward (gradients) | 1× params | 113.8M | 170.5M |
| ADAM `m`, `v` | optimizer (moments) | 2× params | 227.6M | 340.9M |
| **total training floats** | — | **4× params** | **455.2M (~1.8 GB @ fp32)** | **681.8M (~2.7 GB @ fp32)** |

## Dependencies

- Python 3.10+
- NumPy
- JAX (optional, only for `TINY_PRE_TRAIN_BACKEND=jax`)
