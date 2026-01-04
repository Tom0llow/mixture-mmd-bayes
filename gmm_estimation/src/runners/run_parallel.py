from __future__ import annotations

import multiprocessing as mp
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from generator.make_mixture_gaussian import sample_mixture_gaussian
from metrics.ed2_unbiased import ed2_unbiased
from optimizer.mfld_mix_mmd_vi import mfld_mix_mmd_vi
from optimizer.mfld_mmd_vi import mfld_mmd_vi
from optimizer.mfld_mmd_vi_gmm1 import mfld_mmd_vi_gmm1
from sampling.predictive import sample_predictive_particles
from tqdm import tqdm


def _run_method_loop(
    method: str,
    device_idx: int,
    runs: int,
    base_seed: int,
    dim: int,
    K: int,
    separation: float,
    weights: Optional[np.ndarray],
    sigma: float,
    n_train: int,
    n_test: int,
    steps: int,
    M: int,
    S: int,
    gamma_scale: float,
    prior_tau2: float,
    lambda_kl: float,
    out_queue: mp.Queue[Dict[str, Any]],
) -> None:
    """Worker run: perform `runs` repetitions of single method on specified GPU.

    Puts result dict {"type": "progress"|"result"|"error", ...} to `out_queue`.
    """
    try:
        # Ensure each process uses its assigned CUDA device index
        device = f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(device_idx)
            except Exception:
                pass

        dtype = torch.float64

        beta = float(n_train * dim)
        gamma = gamma_scale * beta

        vals: List[float] = []
        for i in range(runs):
            seed = int(base_seed + device_idx * 1000000 + i)
            rng = np.random.default_rng(seed)

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

            if method == "MMD":
                _, _, theta_particles = mfld_mmd_vi(
                    X_train,
                    beta=beta,
                    steps=steps,
                    lr=5e-3,
                    M=M,
                    S=S,
                    sigma_x2=sigma**2,
                    prior_tau2=prior_tau2,
                    lambda_kl=lambda_kl,
                    device=device,
                    dtype=dtype,
                    return_particles=True,
                )
                Y = sample_predictive_particles(
                    theta_particles, m=n_test, sigma=sigma, device=device, dtype=dtype
                )

            elif method == "GMM-MMD":
                _, _, theta_particles = mfld_mmd_vi_gmm1(
                    X_train,
                    beta=beta,
                    steps=steps,
                    lr=5e-3,
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
                Y = sample_predictive_particles(
                    theta_particles, m=n_test, sigma=sigma, device=device, dtype=dtype
                )

            elif method == "Mixture-MMD":
                _, _, theta_particles = mfld_mix_mmd_vi(
                    X_train,
                    beta=beta,
                    gamma=gamma,
                    steps=steps,
                    lr=5e-3,
                    M=M,
                    S=S,
                    sigma_x2=sigma**2,
                    prior_tau2=prior_tau2,
                    lambda_kl=lambda_kl,
                    device=device,
                    dtype=dtype,
                    return_particles=True,
                )
                Y = sample_predictive_particles(
                    theta_particles, m=n_test, sigma=sigma, device=device, dtype=dtype
                )

            else:
                raise ValueError(f"Unknown method: {method}")

            ed2 = float(ed2_unbiased(X_test, Y).item())
            vals.append(ed2)

            # Send progress update to main process
            out_queue.put({"type": "progress", "method": method, "device": device_idx})

        out_queue.put(
            {"type": "result", "method": method, "device": device_idx, "vals": vals}
        )
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        out_queue.put(
            {
                "type": "error",
                "method": method,
                "device": device_idx,
                "error": str(e),
                "traceback": tb,
                "vals": [],
            }
        )


def run_parallel(
    devices: List[int],
    R: int = 100,
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
    prior_tau2: float = 4.0,
    lambda_kl: float = 0.005,
    base_seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Dispatch processes across GPUs and methods, aggregate ED^2 results.

    Returns dict keyed by method with mean, stderr, runs.
    """
    if len(devices) < 2:
        raise RuntimeError("At least 2 GPUs required for run_parallel")

    methods = ["MMD", "GMM-MMD", "Mixture-MMD"]

    # Distribute R runs of each method across available GPUs
    # Each method gets exactly R runs, spread across devices
    num_devices: int = len(devices)

    ctx = mp.get_context("spawn")
    q: mp.Queue[Dict[str, Any]] = ctx.Queue()
    procs: List[Any] = []

    # For each method, create processes to run it R times across all devices
    for method_idx, method in enumerate(methods):
        # Distribute R runs of this method across available GPUs
        base_runs = R // num_devices
        rem_runs = R % num_devices

        for device_idx, device in enumerate(devices):
            # First rem_runs devices get one extra run
            runs_for_device = base_runs + (1 if device_idx < rem_runs else 0)

            if runs_for_device > 0:
                p = ctx.Process(
                    target=_run_method_loop,
                    args=(
                        method,
                        device,
                        runs_for_device,
                        base_seed,
                        dim,
                        K,
                        separation,
                        weights,
                        sigma,
                        n_train,
                        n_test,
                        steps,
                        M,
                        S,
                        gamma_scale,
                        prior_tau2,
                        lambda_kl,
                        q,
                    ),
                )
                p.start()
                procs.append(p)

    # collect results with progress bars
    collected: Dict[str, List[float]] = {m: [] for m in methods}
    errors: Dict[str, str] = {}

    # Initialize progress bars for each method
    pbar_dict = {m: tqdm(total=R, desc=m, leave=True) for m in methods}

    # Track completed results per method
    results_received = {m: 0 for m in methods}
    total_messages = len(procs)
    received_count = 0

    while received_count < total_messages:
        res = q.get()

        if res.get("type") == "progress":
            method = res["method"]
            pbar_dict[method].update(1)
            results_received[method] += 1

        elif res.get("type") == "result":
            method = res["method"]
            collected[method].extend(res["vals"])
            received_count += 1

        elif res.get("type") == "error":
            method = res["method"]
            errors[
                f"{res['method']}_device_{res['device']}"
            ] = f"{res['error']}\n{res['traceback']}"
            received_count += 1

    # Close progress bars
    for pbar in pbar_dict.values():
        pbar.close()

    for p in procs:
        p.join()

    if errors:
        print("\n" + "=" * 50)
        print("PROCESS ERRORS")
        print("=" * 50)
        for key, msg in errors.items():
            print(f"\n{key}:")
            print(msg)
        print("=" * 50 + "\n")

    out: Dict[str, Dict[str, float]] = {}
    for k, vals in collected.items():
        arr = np.array(vals, dtype=float)
        mean = float(arr.mean()) if arr.size > 0 else float("nan")
        se = (
            float(arr.std(ddof=1) / max(1.0, np.sqrt(max(1, arr.size))))
            if arr.size > 1
            else 0.0
        )
        out[k] = {"mean_ED2": mean, "stderr": se, "runs": int(arr.size)}
    return out
