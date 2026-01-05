from typing import Dict, Optional, Union

import numpy as np
import torch
from generator.make_mixture_gaussian import sample_mixture_gaussian
from metrics.ed2_unbiased import ed2_unbiased
from optimizer.mfld_mix_mmd_vi import mfld_mix_mmd_vi
from optimizer.mfld_mmd_vi import mfld_mmd_vi
from optimizer.mfld_mmd_vi_gmm1 import mfld_mmd_vi_gmm1
from sampling.predictive import sample_predictive_particles
from tqdm import trange


def run_sequantial(
    dim: int = 1,
    K: int = 2,
    separation: float = 3.0,
    weights: Optional[np.ndarray] = None,
    sigma: float = 1.0,
    n_train: int = 1000,
    n_test: int = 2000,
    steps: int = 800,
    M: int = 64,
    S: int = 32,
    gamma_scale: float = 1e-4,
    lr: float = 5.0e-3,
    beta: Optional[Union[float, str]] = None,
    lambda_kl: float = 0.005,
    prior_tau2: float = 4.0,
    R: int = 100,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float64,
) -> Dict[str, Dict[str, float]]:
    """Run the GMM distribution-fit experiment with ED^2 as the metric.

    Returns:
        out: dict keyed by method name, containing mean ED^2 and standard error.
    """
    if seed is not None:
        base_rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        base_rng = np.random.default_rng()

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Determine numeric beta value: if beta is None or "default", compute
    # automatically as n_train * dim; otherwise coerce provided value to
    # float. Use `beta_val` (float) for subsequent computations to satisfy
    # static type checking.
    if beta is None or beta == "default":
        beta_val = float(n_train * dim)
    else:
        beta_val = float(beta)
    gamma = gamma_scale * beta_val

    results: dict[str, list[float]] = {"MMD": [], "GMM-MMD": [], "Mixture-MMD": []}

    for _ in trange(R, leave=False):
        rng = np.random.default_rng(base_rng.integers(0, 1 << 31))

        X_train_np = sample_mixture_gaussian(
            n=n_train,
            d=dim,
            K=K,
            sigma=sigma,
            separation=separation,
            weights=weights,
            rng=rng,
        )
        X_test_np = sample_mixture_gaussian(
            n=n_test,
            d=dim,
            K=K,
            sigma=sigma,
            separation=separation,
            weights=weights,
            rng=rng,
        )
        X_train = torch.from_numpy(X_train_np).to(device=device, dtype=dtype)
        X_test = torch.from_numpy(X_test_np).to(device=device, dtype=dtype)

        # MMD-Bayes VI
        _, _, theta_particles = mfld_mmd_vi(
            X_train,
            beta=beta_val,
            steps=steps,
            lr=lr,
            M=M,
            S=S,
            sigma_x2=sigma**2,
            prior_tau2=prior_tau2,
            lambda_kl=lambda_kl,
            device=device,
            dtype=dtype,
            return_particles=True,
        )
        Y_mmd = sample_predictive_particles(
            theta_particles, m=n_test, sigma=sigma, device=device, dtype=dtype
        )
        results["MMD"].append(ed2_unbiased(X_test, Y_mmd).item())

        # MMD-Bayes VI with GMM generator
        _, _, theta_particles1 = mfld_mmd_vi_gmm1(
            X_train,
            beta=beta_val,
            steps=steps,
            lr=lr,
            M=1,
            C=M,
            S=S,
            sigma_x2=sigma**2,
            prior_tau2=prior_tau2,
            lambda_kl=lambda_kl,
            device=device,
            dtype=dtype,
            return_particles=True,
        )
        Y_gmm = sample_predictive_particles(
            theta_particles1, m=n_test, sigma=sigma, device=device, dtype=dtype
        )
        results["GMM-MMD"].append(ed2_unbiased(X_test, Y_gmm).item())

        # Mixture MMD-Bayes VI
        _, _, theta_particles2 = mfld_mix_mmd_vi(
            X_train,
            beta=beta_val,
            gamma=gamma,
            steps=steps,
            lr=lr,
            M=M,
            S=S,
            sigma_x2=sigma**2,
            prior_tau2=prior_tau2,
            lambda_kl=lambda_kl,
            device=device,
            dtype=dtype,
            return_particles=True,
        )
        Y_mix = sample_predictive_particles(
            theta_particles2, m=n_test, sigma=sigma, device=device, dtype=dtype
        )
        results["Mixture-MMD"].append(ed2_unbiased(X_test, Y_mix).item())

    out: Dict[str, Dict[str, float]] = {}
    for k, vals in results.items():
        arr = np.array(vals, dtype=float)
        mean = float(arr.mean())
        se = float(arr.std(ddof=1) / max(1.0, np.sqrt(len(arr))))
        out[k] = {"mean_ED2": mean, "stderr": se, "runs": int(len(arr))}
    return out
