from __future__ import annotations

import multiprocessing as mp
import traceback
from typing import Any, Dict, Sequence, Tuple, TypeAlias, Union

import numpy as np
import torch
from numpy.typing import NDArray
from optimizer.mfld_mix_mmd_vi import mfld_mix_mmd_vi
from optimizer.mfld_mmd_vi import mfld_mmd_vi
from optimizer.mfld_mmd_vi_gmm1 import mfld_mmd_vi_gmm1
from sampling.predictive import sample_predictive_particles

Array: TypeAlias = NDArray[np.floating[Any]]

WorkerSuccessMsg = Tuple[str, Array]
WorkerErrorMsg = Tuple[str, None, str, str]
WorkerMsg = Union[WorkerSuccessMsg, WorkerErrorMsg]


def _worker_method_run(
    method: str,
    device_idx: int,
    Xtr_np: Array,
    Xte_np: Array,
    sigma: float,
    steps: int,
    lr: float,
    M: int,
    S: int,
    gamma_scale: float,
    seed: int,
    dtype: torch.dtype,
    out_q: mp.Queue[WorkerMsg],
) -> None:
    try:
        device = (
            f"cuda:{device_idx}"
            if torch.cuda.is_available() and device_idx >= 0
            else "cpu"
        )
        if torch.cuda.is_available() and device_idx >= 0:
            try:
                torch.cuda.set_device(device_idx)
            except Exception:
                pass

        Xtr = torch.from_numpy(Xtr_np).to(device=device, dtype=dtype)
        beta = float(Xtr.shape[0] * Xtr.shape[1])
        gamma = gamma_scale * beta

        if method == "MMD":
            _, _, theta = mfld_mmd_vi(
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
        elif method == "GMM-MMD":
            _, _, theta = mfld_mmd_vi_gmm1(
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
        elif method == "Mixture-MMD":
            _, _, theta = mfld_mix_mmd_vi(
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
        else:
            raise ValueError(f"Unknown method: {method}")

        Y = (
            sample_predictive_particles(
                theta, m=Xte_np.shape[0], sigma=sigma, device=device, dtype=dtype
            )
            .cpu()
            .numpy()
        )
        out_q.put((method, Y))
    except Exception as e:
        tb = traceback.format_exc()
        out_q.put((method, None, str(e), tb))


def run_methods_sequential(
    methods: Sequence[str],
    Xtr_np: Array,
    Xte_np: Array,
    sigma: float,
    steps: int,
    lr: float,
    M: int,
    S: int,
    gamma_scale: float,
    seed: int,
    dtype: torch.dtype,
) -> Dict[str, Array]:
    """Run each method sequentially in the current process and return predictions.

    Returns a dict mapping method name to numpy array of predictive samples.
    """
    out: Dict[str, Array] = {}

    # choose device for sequential runs: prefer CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for method in methods:
        Xtr = torch.from_numpy(Xtr_np).to(device=device, dtype=dtype)
        beta = float(Xtr.shape[0] * Xtr.shape[1])
        gamma = gamma_scale * beta

        if method == "MMD":
            _, _, theta = mfld_mmd_vi(
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
        elif method == "GMM-MMD":
            _, _, theta = mfld_mmd_vi_gmm1(
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
        elif method == "Mixture-MMD":
            _, _, theta = mfld_mix_mmd_vi(
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
        else:
            raise ValueError(f"Unknown method: {method}")

        Y = (
            sample_predictive_particles(
                theta, m=Xte_np.shape[0], sigma=sigma, device=device, dtype=dtype
            )
            .cpu()
            .numpy()
        )
        out[method] = Y

    return out


def run_methods_parallel(
    methods: Sequence[str],
    Xtr_np: Array,
    Xte_np: Array,
    sigma: float,
    steps: int,
    lr: float,
    M: int,
    S: int,
    gamma_scale: float,
    seed: int,
    dtype: torch.dtype,
    devices: Sequence[int],
) -> Dict[str, Array]:
    """Run methods in parallel processes and collect predictive samples.

    Returns a dict mapping method name to numpy array of predictive samples.
    """
    ctx = mp.get_context("spawn")
    q: mp.Queue[WorkerMsg] = ctx.Queue()
    procs = []
    out: Dict[str, Array] = {}

    for i, method in enumerate(methods):
        device_idx = devices[i % len(devices)]
        p = ctx.Process(
            target=_worker_method_run,
            args=(
                method,
                device_idx,
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
                q,
            ),
        )
        p.start()
        procs.append(p)

    received = 0
    while received < len(procs):
        msg = q.get()
        # success tuple: (method, Y)
        if len(msg) >= 2 and msg[1] is not None:
            method, Y = msg[0], msg[1]
            out[method] = Y
        else:
            # error
            method = msg[0]
            err = msg[2] if len(msg) > 2 else "unknown"
            tb = msg[3] if len(msg) > 3 else ""
            raise RuntimeError(f"Plot worker for {method} failed: {err}\n{tb}")
        received += 1

    for p in procs:
        p.join()

    return out
