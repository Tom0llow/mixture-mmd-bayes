from __future__ import annotations

import multiprocessing as mp
from typing import Any, Dict, List

import numpy as np
import torch
from generator.make_mixture_gaussian import sample_mixture_gaussian
from metrics.ed2_unbiased import ed2_unbiased
from optimizer.mfld_mix_mmd_vi import mfld_mix_mmd_vi
from sampling.predictive import sample_predictive_particles
from tqdm import tqdm


def _run_gamma_scale_worker(
    device_idx: int,
    runs: int,
    base_seed: int,
    dim: int,
    K: int,
    separation: float,
    weights: Any,
    means: Any,
    layout_key: str,
    sigma: float,
    n_train: int,
    n_test: int,
    steps: int,
    M: int,
    S: int,
    gamma_scale: float,
    lr: float,
    beta_cfg: Any,
    prior_tau2: float,
    lambda_kl: float,
    out_queue: mp.Queue[Dict[str, Any]],
) -> None:
    try:
        device = f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(device_idx)
            except Exception:
                pass

        dtype = torch.float64

        if beta_cfg is None or beta_cfg == "default":
            beta_val = float(n_train * dim)
        else:
            beta_val = float(beta_cfg)
        gamma = gamma_scale * beta_val

        vals: List[float] = []
        for i in range(runs):
            seed = int(base_seed + device_idx * 1_000_000 + i)
            rng = np.random.default_rng(seed)

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

            out_queue.put({"type": "progress", "device": device_idx})

        out_queue.put({"type": "result", "device": device_idx, "vals": vals})
    except Exception as e:
        import traceback

        out_queue.put(
            {
                "type": "error",
                "device": device_idx,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "vals": [],
            }
        )


def run_gamma_sweep_parallel(
    exp_params: Dict[str, Any],
    opt_params: Dict[str, Any],
    gamma_scales: List[float],
    R: int,
    devices: List[int],
    base_seed: int,
) -> List[Dict[str, Any]]:
    """Run gamma sweep in parallel across devices and aggregate ED2 statistics."""
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

    results: List[Dict[str, Any]] = []

    for gamma_scale in gamma_scales:
        ctx = mp.get_context("spawn")
        q: mp.Queue[Dict[str, Any]] = ctx.Queue()
        procs = []

        num_devices = len(devices)
        base_runs = R // num_devices
        rem_runs = R % num_devices

        for device_idx, device in enumerate(devices):
            runs_for_device = base_runs + (1 if device_idx < rem_runs else 0)
            if runs_for_device > 0:
                p = ctx.Process(
                    target=_run_gamma_scale_worker,
                    args=(
                        device,
                        runs_for_device,
                        base_seed,
                        dim,
                        K,
                        separation,
                        weights,
                        means,
                        layout_key,
                        sigma,
                        n_train,
                        n_test,
                        steps,
                        M,
                        S,
                        float(gamma_scale),
                        lr,
                        beta_cfg,
                        prior_tau2,
                        lambda_kl,
                        q,
                    ),
                )
                p.start()
                procs.append(p)

        pbar = tqdm(total=R, desc=f"gamma_scale={gamma_scale}", leave=True)
        collected: List[float] = []
        errors: Dict[str, str] = {}
        received_count = 0
        total_messages = len(procs)

        while received_count < total_messages:
            res = q.get()
            if res.get("type") == "progress":
                pbar.update(1)
            elif res.get("type") == "result":
                collected.extend(res.get("vals", []))
                received_count += 1
            elif res.get("type") == "error":
                errors[f"device_{res['device']}"] = f"{res.get('error')}\n{res.get('traceback')}"
                received_count += 1

        pbar.close()

        for p in procs:
            p.join()

        if errors:
            print("\n" + "=" * 50)
            print(f"PROCESS ERRORS (gamma_scale={gamma_scale})")
            print("=" * 50)
            for key, msg in errors.items():
                print(f"\n{key}:")
                print(msg)
            print("=" * 50 + "\n")

        arr = np.array(collected, dtype=float)
        mean = float(arr.mean()) if arr.size > 0 else float("nan")
        se = float(arr.std(ddof=1) / max(1.0, np.sqrt(max(1, arr.size)))) if arr.size > 1 else 0.0

        results.append(
            {
                "gamma_scale": float(gamma_scale),
                "mean_ED2": mean,
                "stderr": se,
                "runs": int(arr.size),
            }
        )

    return results
