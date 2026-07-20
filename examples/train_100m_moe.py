"""
Train a ~170M-parameter Mixture-of-Experts GPT-2 byte-level on this repo's own
source code. Same recipe as examples/train_100m.py, but each block's dense FFN
is replaced with a 4-expert top-2 MoEFeedForward and the layer count is halved
— more total parameters than the dense 113M model, fewer active per token.
A Switch-style load-balancing aux loss (AUX_COEF, default 0.01) keeps the
routers from collapsing onto 1-2 experts; set AUX_COEF=0 to watch them collapse.
Run (from repo root, JAX float32 backend strongly recommended):
    TINY_PRE_TRAIN_BACKEND=jax TINY_PRE_TRAIN_JAX_X64=0 .venv/bin/python examples/train_100m_moe.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as onp  # bookkeeping only

from models.gpt2 import GPT2
from layers.moe import MoEFeedForward
from losses.losses import SoftmaxCrossEntropy
from optim.adam import ADAM

SEQ_LEN = 128
BATCH = 8
D_MODEL = 768
N_LAYERS = 8
N_EXPERTS = 4
TOP_K = 2
STEPS = int(os.environ.get("TRAIN_STEPS", 150))
AUX_COEF = float(os.environ.get("AUX_COEF", 0.01))  # 0 → routers collapse onto 1-2 experts
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


def n_params(module) -> int:
    return sum(int(onp.prod(p.data.shape)) for p in module.parameters())


def main():
    onp.random.seed(0)
    data = load_corpus()
    print(f"Corpus: {len(data):,} bytes")

    model = GPT2(vocab_size=256, d_model=D_MODEL, n_heads=12, n_layers=N_LAYERS,
                 max_seq_len=256)
    for block in model.blocks:
        block.ffn = MoEFeedForward(D_MODEL, n_experts=N_EXPERTS, top_k=TOP_K,
                                   aux_coef=AUX_COEF)

    total = n_params(model)
    expert = n_params(model.blocks[0].ffn.experts[0])
    inactive_per_block = (N_EXPERTS - TOP_K) * expert
    active = total - N_LAYERS * inactive_per_block
    print(f"Parameters: {total:,} total, {active:,} active per token "
          f"({N_EXPERTS} experts, top-{TOP_K}, {N_LAYERS} layers)")

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
            aux = sum(float(b.ffn.aux_loss) for b in model.blocks)
            print(f"step {step:4d}  loss {float(loss):.4f}  aux {aux:.4f}  "
                  f"({dt / step:.2f}s/step)", flush=True)

    print("\nGate mass per expert (per block):")
    for i, block in enumerate(model.blocks):
        frac = onp.asarray(block.ffn._gates).reshape(-1, N_EXPERTS).mean(axis=0)
        print(f"  block {i}: " + "  ".join(f"{f:.2f}" for f in frac))

    prompt_text = "class Linear(La"
    prompt = onp.frombuffer(prompt_text.encode(), dtype=onp.uint8).astype(onp.int64)[None, :]
    out = model.generate(prompt, max_new_tokens=120, temperature=0.8, top_k=40)
    print("\n--- sample ---")
    print(bytes(onp.asarray(out, dtype=onp.uint8)).decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
