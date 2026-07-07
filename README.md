# tiny_ml

A from-scratch machine learning library built on pure NumPy — no PyTorch, no TensorFlow. Implements the full stack from basic autodiff primitives up to GPT-2 and T5.

## What's implemented

**Layers**
- `Linear` — fully connected layer with He initialization
- Activations: `ReLU`, `Sigmoid`, `Tanh`, `GeLU`
- `LayerNorm` — with learnable `gamma`/`beta`
- `Embedding`, `SinusoidalPositionalEmbedding`, `LearnedPositionalEmbedding`, `FeatureEmbedding`
- `FeedForward` — position-wise FFN with configurable activation and expansion factor
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
