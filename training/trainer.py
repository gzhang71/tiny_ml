import numpy as np

from core.module import Model, Loss, Optimizer
from optim.clip import clip_grad_norm


class Trainer:
    """Training loop: forward → loss → backward → (clip) → step.

    The minimal path is unchanged —
    `Trainer(model, loss_fn, optimizer).fit(x, y, epochs)` — with the machinery
    a real pre-training run needs available as options:

    - `lr_schedule`: callable `step -> lr`, assigned to `optimizer.lr` before
      every step (see `optim/schedule.py`)
    - `max_grad_norm`: global-norm gradient clipping, applied after the
      backward pass and before the optimizer step
    - `grad_accum_steps`: run this many micro-batches per optimizer step, to
      reach an effective batch size larger than memory allows

    `self.step` counts *optimizer* steps, not micro-batches, so a schedule
    written against a total step budget stays correct when accumulation changes.
    """

    def __init__(
        self,
        model: Model,
        loss_fn: Loss,
        optimizer: Optimizer,
        lr_schedule=None,
        max_grad_norm: float | None = None,
        grad_accum_steps: int = 1,
    ):
        assert grad_accum_steps >= 1
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.max_grad_norm = max_grad_norm
        self.grad_accum_steps = grad_accum_steps
        self.step = 0
        self.last_grad_norm: float | None = None

    # ---- single steps ------------------------------------------------------

    def _accumulate(self, x, y, scale: float) -> float:
        """One micro-batch: forward, backward, accumulate grads. No step.

        `scale` divides the loss gradient so that accumulating k micro-batches
        yields the mean gradient over all of them, matching one batch of k×
        the size. Without it, accumulation would silently multiply the
        effective learning rate by k.
        """
        pred = self.model.forward(x)
        loss = self.loss_fn.forward(pred, y)
        self.model.backward(self.loss_fn.backward() * scale)
        return float(loss)

    def _apply_step(self) -> None:
        """Clip, set the scheduled LR, and take one optimizer step."""
        self.step += 1
        if self.max_grad_norm is not None:
            self.last_grad_norm = clip_grad_norm(
                self.optimizer.parameters(), self.max_grad_norm
            )
        if self.lr_schedule is not None:
            self.optimizer.lr = self.lr_schedule(self.step)
        self.optimizer.step()

    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        """One optimizer step on a single batch (no accumulation)."""
        self.optimizer.zero_grad()
        loss = self._accumulate(x, y, 1.0)
        self._apply_step()
        return loss

    def train_step_accumulated(self, micro_batches) -> float:
        """One optimizer step over several micro-batches.

        `micro_batches` is a sequence of (x, y) pairs; its length is the
        accumulation factor for this step. Returns the mean micro-batch loss.
        """
        micro_batches = list(micro_batches)
        self.optimizer.zero_grad()
        scale = 1.0 / len(micro_batches)
        total = sum(self._accumulate(x, y, scale) for x, y in micro_batches)
        self._apply_step()
        return total / len(micro_batches)

    # ---- loops ---------------------------------------------------------------

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int = 32,
        verbose: bool = True,
        val_data: tuple | None = None,
    ) -> dict[str, list[float]]:
        """Train for `epochs` passes over (x, y), shuffling each epoch.

        `val_data=(x_val, y_val)` reports held-out loss after every epoch — the
        only way to see overfitting, which a training curve alone hides.
        Returns the per-epoch history as a dict of lists.
        """
        n = x.shape[0]
        history: dict[str, list[float]] = {"loss": []}
        if val_data is not None:
            history["val_loss"] = []

        self.model.train()
        for epoch in range(1, epochs + 1):
            indices = np.random.permutation(n)
            epoch_loss = 0.0
            num_steps = 0

            # each optimizer step consumes grad_accum_steps micro-batches
            micro = batch_size
            per_step = micro * self.grad_accum_steps
            for start in range(0, n, per_step):
                chunk = indices[start : start + per_step]
                batches = [
                    (x[chunk[i : i + micro]], y[chunk[i : i + micro]])
                    for i in range(0, len(chunk), micro)
                ]
                if not batches:
                    continue
                epoch_loss += (
                    self.train_step(*batches[0])
                    if len(batches) == 1
                    else self.train_step_accumulated(batches)
                )
                num_steps += 1

            mean_loss = epoch_loss / max(num_steps, 1)
            history["loss"].append(mean_loss)

            message = f"Epoch {epoch:3d}/{epochs}  loss={mean_loss:.6f}"
            if val_data is not None:
                val_loss = self.evaluate(*val_data)
                history["val_loss"].append(val_loss)
                message += f"  val_loss={val_loss:.6f}"
            if self.lr_schedule is not None:
                message += f"  lr={self.optimizer.lr:.2e}"
            if verbose:
                print(message)

        return history

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Forward pass in eval mode (dropout off), restoring the prior mode."""
        was_training = self.model.training
        self.model.eval()
        try:
            return self.model.forward(x)
        finally:
            if was_training:
                self.model.train()

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(x)
        return float(self.loss_fn.forward(pred, y))
