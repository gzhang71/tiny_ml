"""
Regression demo: fit a noisy sine wave with an MLP + MSE loss.
Run from the repo root: python -m tiny_ml.examples.regression
"""
import numpy as np
from tiny_ml.models.mlp import MLP
from tiny_ml.losses.losses import MSELoss
from tiny_ml.optim.adam import ADAM
from tiny_ml.training.trainer import Trainer


def main():
    np.random.seed(0)

    # --- data ---
    x = np.linspace(-np.pi, np.pi, 500)[:, None].astype(np.float32)
    y = (np.sin(x) + 0.1 * np.random.randn(*x.shape)).astype(np.float32)

    # --- model ---
    model = MLP([1, 64, 64, 1])
    loss_fn = MSELoss()
    optimizer = ADAM(model.parameters(), lr=1e-3)

    trainer = Trainer(model, loss_fn, optimizer)
    trainer.fit(x, y, epochs=200, batch_size=64)

    pred = trainer.predict(x)
    final_mse = float(np.mean((pred - y) ** 2))
    print(f"\nFinal MSE on training set: {final_mse:.6f}")


if __name__ == "__main__":
    main()
