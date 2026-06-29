"""
Variational Autoencoder (VAE).

Architecture:
  encoder: MLP → (μ head, log σ² head)
  reparameterize: z = μ + ε·exp(0.5·log σ²),  ε ~ N(0, I)
  decoder: MLP → reconstruction

Training loss: ELBO = reconstruction_loss + β·KL(q(z|x) ‖ p(z))

Usage:
    vae = VAE(input_dim=784, hidden_dims=[256, 128], latent_dim=32)
    recon = vae.forward(x)
    recon_loss = MSELoss().forward(recon, x)
    total_loss = recon_loss + beta * vae.kl_loss()
    recon_grad = MSELoss().backward()
    vae.backward(recon_grad, beta=beta)
    optimizer.step()
"""
import numpy as np
from core.module import Model
from layers.linear import Linear
from models.mlp import MLP


class VAE(Model):
    """Variational Autoencoder with MLP encoder and decoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        decoder_hidden_dims: list[int] | None = None,
        activation_cls=None,
    ):
        from layers.activations import ReLU
        activation_cls = activation_cls or ReLU
        dec_dims = decoder_hidden_dims if decoder_hidden_dims is not None else hidden_dims[::-1]

        self.encoder = MLP([input_dim] + hidden_dims, activation=activation_cls)
        self.mu_head = Linear(hidden_dims[-1], latent_dim)
        self.log_var_head = Linear(hidden_dims[-1], latent_dim)
        self.decoder = MLP([latent_dim] + dec_dims + [input_dim], activation=activation_cls)

        self._mu: np.ndarray | None = None
        self._log_var: np.ndarray | None = None
        self._eps: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = self.encoder.forward(x)
        self._mu = self.mu_head.forward(h)
        self._log_var = self.log_var_head.forward(h)
        z = self._reparameterize(self._mu, self._log_var)
        return self.decoder.forward(z)

    def backward(self, grad_recon: np.ndarray, beta: float = 1.0) -> None:
        """Backprop through reconstruction path + KL term.

        grad_recon: dL_recon / d_reconstruction  (from your reconstruction loss)
        beta: weight on the KL term (β-VAE; use 1.0 for standard VAE)
        """
        d_z = self.decoder.backward(grad_recon)

        std = np.exp(0.5 * self._log_var)
        d_mu = d_z.copy()
        d_log_var = d_z * self._eps * std * 0.5

        n = self._mu.shape[0]
        d_mu += beta * self._mu / n
        d_log_var += beta * 0.5 * (np.exp(self._log_var) - 1.0) / n

        d_h = self.mu_head.backward(d_mu) + self.log_var_head.backward(d_log_var)
        self.encoder.backward(d_h)

    def _reparameterize(self, mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
        self._eps = np.random.randn(*mu.shape)
        return mu + self._eps * np.exp(0.5 * log_var)

    def kl_loss(self) -> float:
        """KL divergence KL(q(z|x) ‖ N(0,I)), averaged over the batch."""
        return float(-0.5 * np.mean(1.0 + self._log_var - self._mu ** 2 - np.exp(self._log_var)))

    def encode(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (mu, log_var) without sampling."""
        h = self.encoder.forward(x)
        return self.mu_head.forward(h), self.log_var_head.forward(h)

    def decode(self, z: np.ndarray) -> np.ndarray:
        return self.decoder.forward(z)

    def sample(self, n: int, latent_dim: int | None = None) -> np.ndarray:
        """Generate n samples by decoding random latent vectors."""
        if latent_dim is None:
            latent_dim = self.mu_head.W.data.shape[1]
        return self.decode(np.random.randn(n, latent_dim))

    def parameters(self) -> list:
        return (
            self.encoder.parameters()
            + self.mu_head.parameters()
            + self.log_var_head.parameters()
            + self.decoder.parameters()
        )
