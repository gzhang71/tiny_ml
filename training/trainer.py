import numpy as np
from tiny_ml.core.module import Model, Loss, Optimizer


class Trainer:
    """Minimal training loop: forward → loss → backward → step."""

    def __init__(self, model: Model, loss_fn: Loss, optimizer: Optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        self.optimizer.zero_grad()
        pred = self.model.forward(x)
        loss = self.loss_fn.forward(pred, y)
        grad = self.loss_fn.backward()
        self.model.backward(grad)
        self.optimizer.step()
        return loss

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int = 32,
        verbose: bool = True,
    ):
        n = x.shape[0]
        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(n)
            epoch_loss = 0.0
            num_batches = 0
            for start in range(0, n, batch_size):
                idx = indices[start : start + batch_size]
                epoch_loss += self.train_step(x[idx], y[idx])
                num_batches += 1
            if verbose:
                print(f"Epoch {epoch:3d}/{epochs}  loss={epoch_loss / num_batches:.6f}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.forward(x)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(x)
        return self.loss_fn.forward(pred, y)
