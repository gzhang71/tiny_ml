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
from optim.adam import ADAM

SEQ_LEN = 128
BATCH = 8
STEPS = int(os.environ.get("TRAIN_STEPS", 150))
LOG_EVERY = 5


def load_corpus() -> onp.ndarray:
    root = Path(__file__).resolve().parent.parent
    files = sorted(root.rglob("*.py")) + sorted(root.rglob("*.md"))
    text = "\n\n".join(
        f.read_text() for f in files if ".venv" not in f.parts and "__pycache__" not in f.parts
    )
    return onp.frombuffer(text.encode("utf-8"), dtype=onp.uint8)


def make_batch(data: onp.ndarray):
    starts = onp.random.randint(0, len(data) - SEQ_LEN - 1, size=BATCH)
    x = onp.stack([data[s: s + SEQ_LEN] for s in starts]).astype(onp.int64)
    y = onp.stack([data[s + 1: s + SEQ_LEN + 1] for s in starts]).astype(onp.int64)
    return x, y


def main():
    onp.random.seed(0)
    data = load_corpus()
    print(f"Corpus: {len(data):,} bytes")

    # 16 blocks x ~7.1M + embeddings ≈ 113M parameters
    model = GPT2(vocab_size=256, d_model=768, n_heads=12, n_layers=16, max_seq_len=256)
    n_params = sum(int(onp.prod(p.data.shape)) for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    loss_fn = SoftmaxCrossEntropy()
    optimizer = ADAM(model.parameters(), lr=3e-4)

    t0 = time.time()
    for step in range(1, STEPS + 1):
        x, y = make_batch(data)
        optimizer.zero_grad()
        logits = model.forward(x)
        loss = loss_fn.forward(logits.reshape(-1, 256), y.reshape(-1))
        model.backward(loss_fn.backward().reshape(BATCH, SEQ_LEN, 256))
        optimizer.step()
        if step % LOG_EVERY == 0 or step == 1:
            dt = time.time() - t0
            print(f"step {step:4d}  loss {float(loss):.4f}  ({dt / step:.2f}s/step)", flush=True)

    prompt_text = "class Linear(La"
    prompt = onp.frombuffer(prompt_text.encode(), dtype=onp.uint8).astype(onp.int64)[None, :]
    out = model.generate(prompt, max_new_tokens=120, temperature=0.8, top_k=40)
    print("\n--- sample ---")
    print(bytes(onp.asarray(out, dtype=onp.uint8)).decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
