"""
Train a ~113M-parameter GPT-2 style model byte-level on this repo's own source code.
Run (from repo root, JAX float32 backend strongly recommended):
    TINY_PRE_TRAIN_BACKEND=jax TINY_PRE_TRAIN_JAX_X64=0 .venv/bin/python examples/train_100m.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as onp  # bookkeeping only

from models.gpt2 import GPT2
from losses.losses import SoftmaxCrossEntropy
from optim.adamw import AdamW, decay_groups
from optim.clip import clip_grad_norm
from optim.schedule import CosineWithWarmup
from training import checkpoint

SEQ_LEN = 128
BATCH = 8
STEPS = int(os.environ.get("TRAIN_STEPS", 150))
LOG_EVERY = 5
WARMUP_STEPS = max(1, STEPS // 20)
MAX_GRAD_NORM = 1.0
CHECKPOINT_EVERY = int(os.environ.get("CHECKPOINT_EVERY", 0))  # 0 = off
VAL_FRACTION = 0.05


def load_corpus() -> onp.ndarray:
    root = Path(__file__).resolve().parent.parent
    files = sorted(root.rglob("*.py")) + sorted(root.rglob("*.md"))
    text = "\n\n".join(
        f.read_text() for f in files if ".venv" not in f.parts and "__pycache__" not in f.parts
    )
    return onp.frombuffer(text.encode("utf-8"), dtype=onp.uint8)


def split_corpus(data: onp.ndarray) -> tuple[onp.ndarray, onp.ndarray]:
    """Hold out a contiguous tail for validation.

    Contiguous rather than random windows: sampling val windows from the middle
    of the training text would overlap training windows almost everywhere and
    report a validation loss that is really a training loss.
    """
    split = int(len(data) * (1.0 - VAL_FRACTION))
    return data[:split], data[split:]


def make_batch(data: onp.ndarray, rng: onp.random.RandomState | None = None):
    draw = rng if rng is not None else onp.random
    starts = draw.randint(0, len(data) - SEQ_LEN - 1, size=BATCH)
    x = onp.stack([data[s: s + SEQ_LEN] for s in starts]).astype(onp.int64)
    y = onp.stack([data[s + 1: s + SEQ_LEN + 1] for s in starts]).astype(onp.int64)
    return x, y


def estimate_val_loss(model, loss_fn, val_data, batches: int = 5) -> float:
    """Mean loss over a few fixed validation batches (same batches every call)."""
    rng = onp.random.RandomState(1234)  # fixed: comparable across steps
    model.eval()
    total = 0.0
    for _ in range(batches):
        x, y = make_batch(val_data, rng)
        logits = model.forward(x)
        total += float(loss_fn.forward(logits.reshape(-1, 256), y.reshape(-1)))
    model.train()
    return total / batches


def main():
    onp.random.seed(0)
    train_data, val_data = split_corpus(load_corpus())
    print(f"Corpus: {len(train_data):,} train / {len(val_data):,} val bytes")

    # 16 blocks x ~7.1M + embeddings ≈ 113M parameters
    model = GPT2(vocab_size=256, d_model=768, n_heads=12, n_layers=16, max_seq_len=256)
    n_params = sum(int(onp.prod(p.data.shape)) for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    loss_fn = SoftmaxCrossEntropy()
    params = model.parameters()
    # decay the matmuls, not the biases and LayerNorm gains
    _, no_decay = decay_groups(model)
    optimizer = AdamW(params, lr=3e-4, weight_decay=0.1, no_decay=no_decay)
    schedule = CosineWithWarmup(peak_lr=3e-4, warmup_steps=WARMUP_STEPS,
                                total_steps=STEPS, min_lr=3e-5)

    t0 = time.time()
    for step in range(1, STEPS + 1):
        x, y = make_batch(train_data)
        optimizer.zero_grad()
        logits = model.forward(x)
        loss = loss_fn.forward(logits.reshape(-1, 256), y.reshape(-1))
        model.backward(loss_fn.backward().reshape(BATCH, SEQ_LEN, 256))

        # clip before stepping; a spiking grad norm is the earliest warning
        # sign of divergence, well before the loss shows it
        grad_norm = clip_grad_norm(params, MAX_GRAD_NORM)
        optimizer.lr = schedule(step)
        optimizer.step()

        if step % LOG_EVERY == 0 or step == 1:
            dt = time.time() - t0
            print(
                f"step {step:4d}  loss {float(loss):.4f}  "
                f"gnorm {grad_norm:7.3f}  lr {optimizer.lr:.2e}  "
                f"({dt / step:.2f}s/step)",
                flush=True,
            )
        if CHECKPOINT_EVERY and step % CHECKPOINT_EVERY == 0:
            path = checkpoint.save("gpt2_113m.npz", model, step=step, loss=float(loss))
            print(f"  saved {path} @ step {step}", flush=True)

    val_loss = estimate_val_loss(model, loss_fn, val_data)
    print(f"\nheld-out val loss: {val_loss:.4f}")

    prompt_text = "class Linear(La"
    prompt = onp.frombuffer(prompt_text.encode(), dtype=onp.uint8).astype(onp.int64)[None, :]
    out = model.generate(prompt, max_new_tokens=120, temperature=0.8, top_k=40)
    print("\n--- sample ---")
    print(bytes(onp.asarray(out, dtype=onp.uint8)).decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
