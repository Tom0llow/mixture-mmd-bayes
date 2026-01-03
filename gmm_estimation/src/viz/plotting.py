from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from generator.make_mixture_gaussian import sample_mixture_gaussian
from metrics.ed2_unbiased import ed2_unbiased
from optimizer.mfld_mix_mmd_vi import mfld_mix_mmd_vi
from optimizer.mfld_mmd_vi import mfld_mmd_vi
from optimizer.mfld_mmd_vi_gmm1 import mfld_mmd_vi_gmm1
from sampling.predictive import sample_predictive_particles


def plot_1d(
    dim: int = 1,
    K: int = 2,
    separation: float = 3.0,
    weights: Optional[np.ndarray] = None,
    sigma: float = 1.0,
    n_train: int = 800,
    n_test: int = 2000,
    steps: int = 1200,
    M: int = 32,
    S: int = 32,
    gamma_scale: float = 1e-3,
    seed: int = 0,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    show: bool = True,
    save_dir: Optional[str] = None,
) -> dict[str, float]:
    """1D histogram comparisons for:
        - MMD-Bayes VI
        - GMM-MMD-Bayes VI
        - Mixture-MMD-Bayes VI

    Returns:
        Dictionary of ED^2 values.
    """
    rng = np.random.default_rng(seed)
    Xtr_np = sample_mixture_gaussian(
        n_train, dim, K, sigma=sigma, separation=separation, weights=weights, rng=rng
    )
    Xte_np = sample_mixture_gaussian(
        n_test, dim, K, sigma=sigma, separation=separation, weights=weights, rng=rng
    )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Xtr = torch.from_numpy(Xtr_np).to(device=device, dtype=dtype)
    Xte = torch.from_numpy(Xte_np).to(device=device, dtype=dtype)

    beta = float(n_train * dim)
    gamma = gamma_scale * beta

    _, _, theta_mmd = mfld_mmd_vi(
        Xtr,
        beta=beta,
        steps=steps,
        lr=5e-3,
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
        lr=5e-3,
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
        lr=5e-3,
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
        .ravel()
    )
    Y_gmm = (
        sample_predictive_particles(
            theta_gmm, m=n_test, sigma=sigma, device=device, dtype=dtype
        )
        .cpu()
        .numpy()
        .ravel()
    )
    Y_mix = (
        sample_predictive_particles(
            theta_mix, m=n_test, sigma=sigma, device=device, dtype=dtype
        )
        .cpu()
        .numpy()
        .ravel()
    )
    Xte_1d = Xte.cpu().numpy().ravel()

    def ed2_np(a: np.ndarray, b: np.ndarray) -> float:
        A = torch.tensor(a)[:, None].float()
        B = torch.tensor(b)[:, None].float()
        return float(ed2_unbiased(A, B))

    eds = {
        "MMD-Bayes VI": ed2_np(Xte_1d, Y_mmd),
        "GMM-MMD-Bayes VI": ed2_np(Xte_1d, Y_gmm),
        "Mixture MMD-Bayes VI": ed2_np(Xte_1d, Y_mix),
    }

    for name, Y in [
        ("MMD-Bayes VI", Y_mmd),
        ("GMM-MMD-Bayes VI", Y_gmm),
        ("Mixture MMD-Bayes VI", Y_mix),
    ]:
        plt.figure()
        plt.hist(Xte_1d, bins=60, density=True, alpha=0.4, label="True (test)")
        plt.hist(
            Y, bins=60, density=True, alpha=0.4, label=f"{name} (ED²={eds[name]:.3f})"
        )
        plt.title(f"1D GMM fit: {name} vs True")
        plt.xlabel("x")
        plt.ylabel("density")
        plt.legend()

        if save_dir is not None:
            import os

            os.makedirs(save_dir, exist_ok=True)
            fname = {
                "MMD-Bayes VI": "hist_1d_mmd.png",
                "GMM-MMD-Bayes VI": "hist_1d_gmm.png",
                "Mixture MMD-Bayes VI": "hist_1d_mix.png",
            }[name]
            plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    return eds


def plot_15d(
    K: int = 4,
    separation: float = 3.0,
    weights: Optional[np.ndarray] = None,
    sigma: float = 1.0,
    n_train: int = 2000,
    n_test: int = 5000,
    steps: int = 1200,
    M: int = 64,
    S: int = 64,
    gamma_scale: float = 1e-3,
    seed: int = 1,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
    show: bool = True,
    save_dir: Optional[str] = None,
) -> dict[str, float]:
    """15D PCA(2D) scatter comparisons for:
        - MMD-Bayes VI
        - GMM-MMD-Bayes VI
        - Mixture-MMD-Bayes VI

    Returns:
        Dictionary of ED^2 values.
    """
    rng = np.random.default_rng(seed)
    d = 15
    Xtr_np = sample_mixture_gaussian(
        n_train, d, K, sigma=sigma, separation=separation, weights=weights, rng=rng
    )
    Xte_np = sample_mixture_gaussian(
        n_test, d, K, sigma=sigma, separation=separation, weights=weights, rng=rng
    )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Xtr = torch.from_numpy(Xtr_np).to(device=device, dtype=dtype)
    Xte = torch.from_numpy(Xte_np).to(device=device, dtype=dtype)

    beta = float(n_train * d)
    gamma = gamma_scale * beta

    _, _, theta_mmd = mfld_mmd_vi(
        Xtr,
        beta=beta,
        steps=steps,
        lr=5e-3,
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
        lr=5e-3,
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
        lr=5e-3,
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
    )
    Y_gmm = (
        sample_predictive_particles(
            theta_gmm, m=n_test, sigma=sigma, device=device, dtype=dtype
        )
        .cpu()
        .numpy()
    )
    Y_mix = (
        sample_predictive_particles(
            theta_mix, m=n_test, sigma=sigma, device=device, dtype=dtype
        )
        .cpu()
        .numpy()
    )
    Xte_np = Xte.cpu().numpy()

    # PCA via SVD on centered true test data.
    Xc = Xte_np - Xte_np.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:2].T  # (d, 2)

    P_true = Xc @ W
    P = {
        "MMD-Bayes VI": (Y_mmd - Xte_np.mean(0)) @ W,
        "GMM-MMD-Bayes VI": (Y_gmm - Xte_np.mean(0)) @ W,
        "Mixture-MMD-Bayes VI": (Y_mix - Xte_np.mean(0)) @ W,
    }

    def ed2_np(A: np.ndarray, B: np.ndarray) -> float:
        return float(ed2_unbiased(torch.tensor(A).float(), torch.tensor(B).float()))

    eds = {
        "MMD-Bayes VI": ed2_np(Xte_np, Y_mmd),
        "GMM-MMD-Bayes VI": ed2_np(Xte_np, Y_gmm),
        "Mixture-MMD-Bayes VI": ed2_np(Xte_np, Y_mix),
    }

    for name in ["MMD-Bayes VI", "GMM-MMD-Bayes VI", "Mixture-MMD-Bayes VI"]:
        plt.figure()
        plt.scatter(P_true[:, 0], P_true[:, 1], s=6, alpha=0.3, label="True (test)")
        plt.scatter(
            P[name][:, 0],
            P[name][:, 1],
            s=6,
            alpha=0.3,
            label=f"{name} (ED²={eds[name]:.2f})",
        )
        plt.title(f"15D GMM: PCA(2D) — True vs {name}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()

        if save_dir is not None:
            import os

            os.makedirs(save_dir, exist_ok=True)
            fname = {
                "MMD-Bayes VI": "pca_15d_mmd.png",
                "GMM-MMD-Bayes VI": "pca_15d_gmm.png",
                "Mixture-MMD-Bayes VI": "pca_15d_mix.png",
            }[name]
            plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()

    return eds
