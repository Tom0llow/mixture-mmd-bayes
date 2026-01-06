import math
from typing import Callable, Optional, Tuple

import torch
from core.kernels import gaussian_kernel, median_heuristic_bandwidth
from core.objectives import mix_mmd2, var_emb
from core.simulators import simulate_gaussian_location


def mfld_mix_mmd_vi(
    X: torch.Tensor,
    beta: float,
    gamma: float,
    steps: int = 1000,
    beta_schedule: Callable[[int, int, float], float] = lambda t, T, beta: beta
    * min(1.0, (t + 1) / max(1.0, 0.1 * T)),
    lr: float = 5e-3,
    lr_schedule: Callable[[int, int, float], float] = lambda t, T, lr: lr
    * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * (t - 1) / max(1.0, T - 1)))),
    M: int = 32,
    S: int = 32,
    kernel_fn: Callable[
        [torch.Tensor, torch.Tensor, float], torch.Tensor
    ] = gaussian_kernel,
    simulate_fn: Callable[..., torch.Tensor] = simulate_gaussian_location,
    sigma_x2: float = 1.0,
    prior_tau2: float = 1.0,
    lambda_kl: float = 0.005,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float64,
    return_particles: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Proposed objective optimized via particle updates.

    Returns:
        m_hat: (d,) CPU tensor
        sigma2_hat: (d,) CPU tensor
        theta_particles: (M, d) tensor if return_particles else None
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device, dtype=dtype)
    _, d = X.shape

    bw = median_heuristic_bandwidth(X, max_samples=1000, fallback=float(d))

    theta = (prior_tau2**0.5) * torch.randn(M, d, device=device, dtype=dtype)
    theta.requires_grad_(True)

    for t in range(1, steps + 1):
        beta_t = beta_schedule(t, steps, beta)
        lr_t = lr_schedule(t, steps, lr)

        eps = torch.randn(M, S, d, device=device, dtype=dtype)
        Y = simulate_fn(theta, eps_y=eps, sigma_x2=sigma_x2).reshape(M * S, d)

        mmd_term = mix_mmd2(X, Y, kernel_fn, bw)
        var_term = var_emb(Y, kernel_fn, bw, M, S)
        prior_term = (lambda_kl / (2.0 * prior_tau2)) * theta.pow(2).sum() / M
        U = beta_t * mmd_term + gamma * var_term + prior_term

        grads = torch.autograd.grad(U, theta, create_graph=False)[0]
        with torch.no_grad():
            noise = torch.randn_like(theta) * math.sqrt(2.0 * lambda_kl * lr_t)
            theta.add_(-lr_t * grads + noise)
        theta.requires_grad_(True)

    m_hat = theta.detach().mean(dim=0).cpu()
    sigma2_hat = theta.detach().var(dim=0, unbiased=False).cpu()
    theta_particles = theta.detach() if return_particles else None
    return (m_hat, sigma2_hat, theta_particles)
