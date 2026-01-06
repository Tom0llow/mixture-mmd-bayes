import torch


def simulate_gaussian_location(
    theta: torch.Tensor,
    eps_y: torch.Tensor,
    sigma_x2: float = 1.0,
) -> torch.Tensor:
    """Gaussian location model sampler.

    Args:
        theta: (M, d)
        eps_y: (M, S, d) ~ N(0, I)
        sigma_x2: Observation variance (scalar)

    Returns:
        Y: (M, S, d) with Y = theta[:, None, :] + sqrt(sigma_x2) * eps_y
    """
    return theta[:, None, :] + (sigma_x2**0.5) * eps_y


def simulate_gaussian_mixture_location(
    theta: torch.Tensor,
    eps_y: torch.Tensor,
    sigma_x2: float = 1.0,
) -> torch.Tensor:
    """Gaussian mixture location model sampler (uniform mixture, stratified assignment).

    Args:
        theta:
            (M, C, d) where C is the number of components
        eps_y:
            (M, S, d) ~ N(0, I)
        sigma_x2:
            Shared variance for each component

    Returns:
        Y: (M, S, d)
            For each sample index s, component c = s % C (uniform, stratified),
            and Y[m, s] = theta[m, c] + sqrt(sigma_x2) * eps_y[m, s].
    """
    M, C, d = theta.shape
    _, S, _ = eps_y.shape

    comp_idx = (torch.arange(S, device=theta.device) % C).long()  # (S,)
    comp_idx = comp_idx.unsqueeze(0).expand(M, S)  # (M, S)

    index = comp_idx.unsqueeze(-1).expand(M, S, d)  # (M, S, d)
    means = theta.gather(dim=1, index=index)  # (M, S, d)

    return means + (sigma_x2**0.5) * eps_y
