from typing import Dict, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from generator.make_mixture_gaussian import sample_mixture_gaussian
from metrics.ed2_unbiased import ed2_unbiased
from optimizer.mfld_mix_mmd_vi import mfld_mix_mmd_vi
from optimizer.mfld_mmd_vi import mfld_mmd_vi
from optimizer.mfld_mmd_vi_gmm1 import mfld_mmd_vi_gmm1
from sampling.predictive import sample_predictive_particles
from viz.workers import run_methods_parallel, run_methods_sequential


def plot_1d(
    dim: int = 1,
    K: int = 2,
    separation: float = 3.0,
    weights: Optional[np.ndarray] = None,
    means: Optional[np.ndarray] = None,
    layout_key: str = "auto",
    sigma: float = 1.0,
    n_train: int = 800,
    n_test: int = 2000,
    steps: int = 1200,
    lr: float = 5e-3,
    M: int = 32,
    S: int = 32,
    gamma_scale: float = 1e-3,
    seed: int = 0,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    show: bool = True,
    save_dir: Optional[str] = None,
    mode: str = "sequential",
    devices: Optional[Union[str, Sequence[int]]] = None,
) -> dict[str, float]:
    """Generate 1D histograms comparing predictive samples to test data.

    Returns a dict of ED^2 scores for each method.
    """
    rng = np.random.default_rng(seed)
    Xtr_np = sample_mixture_gaussian(
        n_train,
        dim,
        K,
        sigma=sigma,
        separation=separation,
        weights=weights,
        means=means,
        layout_key=layout_key,
        rng=rng,
    )
    Xte_np = sample_mixture_gaussian(
        n_test,
        dim,
        K,
        sigma=sigma,
        separation=separation,
        weights=weights,
        means=means,
        layout_key=layout_key,
        rng=rng,
    )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    beta = float(n_train * dim)
    gamma = gamma_scale * beta

    methods = ["MMDVI", "MMDVI-GMM", "M-MMDVI"]

    Ys: Dict[str, np.ndarray]
    if mode == "sequential":
        Ys = run_methods_sequential(
            methods,
            Xtr_np,
            Xte_np,
            sigma,
            steps,
            lr,
            M,
            S,
            gamma_scale,
            seed,
            dtype,
        )
    elif mode == "parallel":
        # parse devices
        if devices is None or devices == "auto":
            if torch.cuda.is_available():
                dev_list = list(range(torch.cuda.device_count()))
            else:
                dev_list = [-1]
        elif isinstance(devices, str):
            dev_list = [int(x) for x in devices.split(",") if x.strip()]
        else:
            dev_list = list(devices)

        Ys = run_methods_parallel(
            methods,
            Xtr_np,
            Xte_np,
            sigma,
            steps,
            lr,
            M,
            S,
            gamma_scale,
            seed,
            dtype,
            devices=dev_list,
        )
    else:
        # fallback: local sequential execution
        Xtr = torch.from_numpy(Xtr_np).to(device=device, dtype=dtype)

        _, _, theta_mmd = mfld_mmd_vi(
            Xtr,
            beta=beta,
            steps=steps,
            lr=lr,
            M=M,
            S=S,
            sigma_x2=sigma**2,
            dtype=dtype,
            device=device,
        )
        _, _, theta_gmm = mfld_mmd_vi_gmm1(
            Xtr,
            beta=beta,
            steps=steps,
            lr=lr,
            M=1,
            C=M,
            S=S,
            sigma_x2=sigma**2,
            dtype=dtype,
            device=device,
        )
        _, _, theta_mix = mfld_mix_mmd_vi(
            Xtr,
            beta=beta,
            gamma=gamma,
            steps=steps,
            lr=lr,
            M=M,
            S=S,
            sigma_x2=sigma**2,
            dtype=dtype,
            device=device,
        )

        Y_mmd = (
            sample_predictive_particles(
                theta_mmd, m=n_test, sigma=sigma, device=device, dtype=dtype
            )
            .cpu()
            .numpy()
        ).ravel()
        Y_gmm = (
            sample_predictive_particles(
                theta_gmm, m=n_test, sigma=sigma, device=device, dtype=dtype
            )
            .cpu()
            .numpy()
        ).ravel()
        Y_mix = (
            sample_predictive_particles(
                theta_mix, m=n_test, sigma=sigma, device=device, dtype=dtype
            )
            .cpu()
            .numpy()
        ).ravel()

        Ys = {"MMDVI": Y_mmd, "MMDVI-GMM": Y_gmm, "M-MMDVI": Y_mix}

    Y_mmd = Ys["MMDVI"].ravel()
    Y_gmm = Ys["MMDVI-GMM"].ravel()
    Y_mix = Ys["M-MMDVI"].ravel()

    Xte_1d = Xte_np.ravel()

    def ed2_np(a: np.ndarray, b: np.ndarray) -> float:
        A = torch.tensor(a)[:, None].float()
        B = torch.tensor(b)[:, None].float()
        return float(ed2_unbiased(A, B))

    eds = {
        "MMDVI": ed2_np(Xte_1d, Y_mmd),
        "MMDVI-GMM": ed2_np(Xte_1d, Y_gmm),
        "M-MMDVI": ed2_np(Xte_1d, Y_mix),
    }

    for name, Y in [
        ("MMDVI", Y_mmd),
        ("MMDVI-GMM", Y_gmm),
        ("M-MMDVI", Y_mix),
    ]:
        plt.figure()
        plt.hist(Xte_1d, bins=60, density=True, alpha=0.4, label="True (test)")
        plt.hist(Y, bins=60, density=True, alpha=0.4, label=f"{name}")
        plt.title(f"1D GMM fit: {name} vs True")
        plt.xlabel("x")
        plt.ylabel("density")
        plt.legend()

        if save_dir is not None:
            import os

            os.makedirs(save_dir, exist_ok=True)
            fname = {
                "MMDVI": "hist_1d_mmd.png",
                "MMDVI-GMM": "hist_1d_gmm.png",
                "M-MMDVI": "hist_1d_mix.png",
            }[name]
            plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

        if show:
            plt.show()
        else:
            plt.close()

    return eds
