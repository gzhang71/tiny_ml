# tiny_ml

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
- `ResidualBlock` — two-layer MLP-style residual block
- `MultiHeadAttention` — scaled dot-product with optional causal mask
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
- Losses: `MSELoss`, `SoftmaxCrossEntropy`, `BinaryCrossEntropy`
- Optimizers: `SGD`, `Momentum`, `ADAM`
- Metrics: `precision`, `recall`, `f1_score`, `accuracy`
- `Trainer` — batched fit/predict/evaluate loop

## Running examples

Each model has a self-contained example in `examples/`. Run from the parent directory of this repo:

```bash
python -m tiny_ml.examples.mlp          # spiral classification
python -m tiny_ml.examples.sequential   # sine wave regression
python -m tiny_ml.examples.resnet       # checkerboard classification
python -m tiny_ml.examples.transformer  # next-token prediction
python -m tiny_ml.examples.gpt2         # token generation
python -m tiny_ml.examples.t5           # seq2seq copy task
python -m tiny_ml.examples.vae          # 2D cluster reconstruction
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

If you need more control than `fit()` gives you (custom schedules, gradient inspection, multi-input models like T5), drop down to `trainer.train_step(x, y)` per batch, or write the five-line loop yourself — see the next section. For the generative models, skip `Trainer` for inference and call `model.generate(...)` directly (see `examples/gpt2.py` and `examples/t5.py`).

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
tiny_ml/
├── core/                  # foundations everything else builds on
│   ├── parameter.py       #   Parameter — numpy array + .grad field (the only weight leaf)
│   ├── module.py          #   Module, Layer, Model, Loss, Optimizer base classes
│   └── backend.py         #   array backend (numpy or jax.numpy via TINY_ML_BACKEND)
├── layers/                # reusable building blocks
│   ├── linear.py          #   Linear
│   ├── activations.py     #   ReLU, Sigmoid, Tanh, SiLU, GeLU
│   ├── normalization.py   #   LayerNorm, RMSNorm
│   ├── embedding.py       #   Embedding + positional/feature variants
│   ├── feedforward.py     #   FeedForward, SwiGLU (position-wise FFNs)
│   ├── residual.py        #   ResidualBlock
│   └── attention.py       #   MultiHeadAttention, TransformerBlock, T5/cross attention
├── models/                # full models composed from layers
│   ├── mlp.py             #   MLP
│   ├── sequential.py      #   Sequential
│   ├── resnet.py          #   ResNet
│   ├── transformer.py     #   Transformer (GPT-style decoder-only)
│   ├── gpt2.py            #   GPT2 (learned positions, weight tying, generate())
│   ├── t5.py              #   T5 (encoder-decoder)
│   └── vae.py             #   VAE
├── losses/                # MSELoss, SoftmaxCrossEntropy, BinaryCrossEntropy
├── optim/                 # SGD, Momentum, ADAM
├── metrics/               # precision, recall, f1_score, accuracy
├── training/              # Trainer (fit / predict / evaluate loop)
└── examples/              # one runnable script per model
```

**Class hierarchy.** `Module` is the root: its `parameters()` recurses through `__dict__`, collecting every `Parameter` and nested `Module`, so composition alone wires up the parameter tree. `Layer` and `Model` subclass `Module` purely for naming — layers are building blocks, models are top-level compositions. `Loss` is a separate hierarchy (`forward` returns a scalar, `backward` returns `d_loss/d_pred` from stored state), and `Optimizer` takes the flat parameter list and mutates `.data` in `step()`.

**Data flow.** Each layer stores whatever `forward` computed that `backward` needs in `self._<name>` attributes — there is no tape, so calling `forward` twice before `backward` overwrites that state. Gradients flow top-down: the loss produces the initial gradient, each module's `backward` populates `.grad` on its own parameters and returns the gradient for its input.

**Backend isolation.** Library code never imports numpy directly; everything goes through `core/backend.py` (`from core.backend import xp as np`), which is what lets the same code run on numpy or JAX. Examples, `Trainer`, and metrics use plain numpy since they only do bookkeeping.

## JAX backend (optional speedup)

The library runs on numpy by default. Set `TINY_ML_BACKEND=jax` to route every array
operation through `jax.numpy`/XLA instead — no code changes needed:

```bash
TINY_ML_BACKEND=jax python -m tiny_ml.examples.gpt2      # float64, matches numpy exactly
TINY_ML_BACKEND=jax TINY_ML_JAX_X64=0 python -m tiny_ml.examples.gpt2  # float32, fastest
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

## Dependencies

- Python 3.10+
- NumPy
- JAX (optional, only for `TINY_ML_BACKEND=jax`)
