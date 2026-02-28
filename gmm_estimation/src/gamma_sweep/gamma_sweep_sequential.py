from typing import Any, Dict, List

import numpy as np
import torch
from generator.make_mixture_gaussian import sample_mixture_gaussian
from metrics.ed2_unbiased import ed2_unbiased
from optimizer.mfld_mix_mmd_vi import mfld_mix_mmd_vi
from sampling.predictive import sample_predictive_particles
from tqdm import trange


def run_gamma_sweep_sequential(
    exp_params: Dict[str, Any],
    opt_params: Dict[str, Any],
    gamma_scales: List[float],
    R: int,
) -> List[Dict[str, Any]]:
    """Run gamma sweep sequentially and return per-scale ED2 summary stats."""
    dim = exp_params.get("dim", 1)
    K = exp_params.get("K", 2)
    separation = exp_params.get("separation", 3.0)
    weights = exp_params.get("weights", None)
    means = exp_params.get("means", None)
    layout_key = exp_params.get("layout_key", "auto")
    sigma = exp_params.get("sigma", 1.0)
    n_train = exp_params.get("n_train", 1000)
    n_test = exp_params.get("n_test", 2000)

    steps = opt_params.get("steps", 800)
    M = opt_params.get("M", 32)
    S = opt_params.get("S", 32)
    lr = opt_params.get("lr", 5.0e-3)
    beta_cfg = opt_params.get("beta")
    lambda_kl = opt_params.get("lambda_kl", 0.005)
    prior_tau2 = opt_params.get("prior_tau2", 4.0)

    seed = exp_params.get("seed", None)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        base_rng = np.random.default_rng(seed)
    else:
        base_rng = np.random.default_rng()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    if beta_cfg is None or beta_cfg == "default":
        beta_val = float(n_train * dim)
    else:
        beta_val = float(beta_cfg)

    trial_seeds = base_rng.integers(0, 1 << 31, size=R, dtype=np.int64)

    results: List[Dict[str, Any]] = []

    for gamma_scale in gamma_scales:
        gamma = float(gamma_scale) * beta_val
        vals: List[float] = []

        for i in trange(R, desc=f"gamma_scale={gamma_scale}", leave=False):
            rng = np.random.default_rng(int(trial_seeds[i]))

            X_train_np = sample_mixture_gaussian(
                n=n_train,
                d=dim,
                K=K,
                sigma=sigma,
                separation=separation,
                weights=weights,
                means=means,
                layout_key=layout_key,
                rng=rng,
            )
            X_test_np = sample_mixture_gaussian(
                n=n_test,
                d=dim,
                K=K,
                sigma=sigma,
                separation=separation,
                weights=weights,
                means=means,
                layout_key=layout_key,
                rng=rng,
            )

            X_train = torch.from_numpy(X_train_np).to(device=device, dtype=dtype)
            X_test = torch.from_numpy(X_test_np).to(device=device, dtype=dtype)

            _, _, theta_particles = mfld_mix_mmd_vi(
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
                theta_particles, m=n_test, sigma=sigma, device=device, dtype=dtype
            )
            vals.append(float(ed2_unbiased(X_test, Y_mix).item()))

        arr = np.array(vals, dtype=float)
        mean = float(arr.mean())
        se = float(arr.std(ddof=1) / max(1.0, np.sqrt(len(arr))))
        results.append(
            {
                "gamma_scale": float(gamma_scale),
                "mean_ED2": mean,
                "stderr": se,
                "runs": int(len(arr)),
            }
        )

    return results
