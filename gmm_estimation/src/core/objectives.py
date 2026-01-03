from typing import Callable

import torch


def E_mmd2(
    X: torch.Tensor,
    Y: torch.Tensor,
    kernel_fn: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor],
    bw: float,
    M: int,
    S: int,
) -> torch.Tensor:
    """Existing-method MMD^2 (biased estimator for training stability):
        (1/M) * sum_i MMD^2(P_{theta_i}, P_hat_n)

    Args:
        X: (n, d) data
        Y: (M*S, d) pooled samples
        kernel_fn: kernel(a,b,bw) -> (N,M)
        bw: bandwidth
        M: number of particles
        S: number of simulator samples per particle

    Returns:
        Scalar tensor (mean over particles)
    """
    n, _ = X.shape
    Kxx = kernel_fn(X, X, bw).mean()
    Kxy = kernel_fn(X, Y, bw).view(n, M, S).mean(dim=(0, 2))  # per-particle
    Kyy_blocks = kernel_fn(Y, Y, bw).view(M, S, M, S)
    Kyy_diag = Kyy_blocks.diagonal(dim1=0, dim2=2).permute(2, 0, 1)  # (M, S, S)
    Kyy = Kyy_diag.mean(dim=(1, 2))  # per-particle
    mmd2_per_particle = Kxx - 2.0 * Kxy + Kyy
    return mmd2_per_particle.mean()


def mix_mmd2(
    X: torch.Tensor,
    Y: torch.Tensor,
    kernel_fn: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor],
    bw: float,
) -> torch.Tensor:
    """Mixture MMD^2 computed between data X and pooled samples Y.

    Args:
        X: (n, d)
        Y: (m, d)
        kernel_fn: kernel(a,b,bw)
        bw: bandwidth

    Returns:
        Scalar tensor
    """
    Kxx = kernel_fn(X, X, bw).mean()
    Kxy = kernel_fn(X, Y, bw).mean()
    Kyy = kernel_fn(Y, Y, bw).mean()
    return Kxx - 2.0 * Kxy + Kyy


def var_emb(
    Y: torch.Tensor,
    kernel_fn: Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor],
    bw: float,
    M: int,
    S: int,
) -> torch.Tensor:
    """RKHS embedding variance term (equals E_theta MMD^2(P_theta, mix)).

    Args:
        Y: (M*S, d) pooled samples
        kernel_fn: kernel(a,b,bw)
        bw: bandwidth
        M: number of particles
        S: number of samples per particle

    Returns:
        Scalar tensor
    """
    K = kernel_fn(Y, Y, bw)  # (M*S, M*S)
    pooled = K.mean()  # E_mix[k]
    K_blocks = K.view(M, S, M, S)
    within = K_blocks.diagonal(dim1=0, dim2=2).mean()  # E_theta E_{Y,Y'~P_theta}[k]
    return within - pooled
