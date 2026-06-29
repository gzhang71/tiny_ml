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

## Dependencies

- Python 3.10+
- NumPy
