from typing import Optional

import torch


def sample_predictive_particles(
    theta: torch.Tensor,
    m: int,
    sigma: float,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Predictive sampling from a particle mixture by selecting particles uniformly.

    Args:
        theta: (M, d) particle locations
        m: number of predictive samples
        sigma: observation std
        device: device for computation (defaults to theta.device)
        dtype: output dtype

    Returns:
        Y: (m, d)
    """
    device = device or theta.device
    M, d = theta.shape
    idx = torch.randint(low=0, high=M, size=(m,), device=device)
    eps = torch.randn(m, d, device=device, dtype=dtype)
    return theta[idx, :] + sigma * eps
