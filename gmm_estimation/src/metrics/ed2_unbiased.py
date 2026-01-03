import torch


def ed2_unbiased(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Unbiased U-statistic estimator of Energy Distance^2 between two samples.

    Args:
        X: (n, d)
        Y: (m, d)

    Returns:
        Scalar tensor
    """
    n, m = X.shape[0], Y.shape[0]
    if n < 2 or m < 2:
        # Fallback to a biased variant when sample sizes are too small.
        xy = torch.cdist(X, Y).mean() * 2.0
        xx = torch.cdist(X, X).mean()
        yy = torch.cdist(Y, Y).mean()
        return xy - xx - yy

    xy = torch.cdist(X, Y).mean() * 2.0
    xx_full = torch.cdist(X, X)
    yy_full = torch.cdist(Y, Y)
    xx = (xx_full.sum() - torch.diagonal(xx_full, 0).sum()) / (n * (n - 1))
    yy = (yy_full.sum() - torch.diagonal(yy_full, 0).sum()) / (m * (m - 1))
    return xy - xx - yy
