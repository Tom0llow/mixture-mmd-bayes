import torch


def gaussian_kernel(a: torch.Tensor, b: torch.Tensor, bw: float) -> torch.Tensor:
    """RBF kernel k(x, y) = exp(-||x - y||^2 / bw)

    Args:
        a: Tensor of shape (N, d)
        b: Tensor of shape (M, d)
        bw: Bandwidth (positive scalar)

    Returns:
        Kernel matrix of shape (N, M)
    """
    return torch.exp(-(torch.cdist(a, b) ** 2) / bw)


@torch.no_grad()  # type: ignore[misc]
def median_heuristic_bandwidth(
    X: torch.Tensor,
    max_samples: int = 1000,
    fallback: float | None = None,
) -> float:
    """Median heuristic bandwidth: median(||xi-xj||)^2 using up to max_samples points.

    Args:
        X: (n, d)
        max_samples: Subsample size for pairwise distance computation
        fallback: If distances degenerate, use this value; if None, use d

    Returns:
        Bandwidth (float)
    """
    n, d = X.shape
    device = X.device
    idx = torch.randperm(n, device=device)[: min(max_samples, n)]
    pair = torch.pdist(X[idx])

    if pair.numel() == 0:
        return float(fallback if fallback is not None else d)

    bw = pair.median().pow(2).item()
    if bw < 1e-12:
        bw = float(fallback if fallback is not None else d)
    return float(bw)
