from pathlib import Path
from typing import Any, Dict

import torch
from gamma_sweep.gamma_sweep_parallel import run_gamma_sweep_parallel
from gamma_sweep.gamma_sweep_sequential import run_gamma_sweep_sequential
from utils.output import save_gamma_sweep_results
from viz.plot_gamma_sweep import plot_gamma_sweep


def run_gamma_sweep(
    sweep_config: Dict[str, Any],
    exp_config: Dict[str, Any],
    opt_config: Dict[str, Any],
    runner_config: Dict[str, Any],
    results_root: Path,
) -> Path:
    """Run a gamma-scale sweep and save aggregated metrics and plots."""
    scenario_value = sweep_config.get("scenario", None)
    if isinstance(scenario_value, str):
        scenario = scenario_value
    elif isinstance(scenario_value, list):
        if len(scenario_value) != 1:
            raise ValueError("gamma_sweep.yaml 'scenario' must be a single name or a 1-item list")
        scenario = str(scenario_value[0])
    else:
        raise ValueError("gamma_sweep.yaml 'scenario' must be a string or a 1-item list")

    if scenario not in exp_config:
        raise ValueError(f"Experiment scenario '{scenario}' not found in experiments.yaml")

    exp_params = exp_config[scenario]

    if scenario in opt_config:
        opt_params = opt_config[scenario]
        active_optimizer = scenario
    else:
        named = exp_params.get("optimizer", None)
        if named and named in opt_config:
            opt_params = opt_config[named]
            active_optimizer = named
        else:
            global_default_name = opt_config.get("optimizer", "default")
            if global_default_name in opt_config:
                opt_params = opt_config[global_default_name]
                active_optimizer = global_default_name
            elif "default" in opt_config:
                opt_params = opt_config["default"]
                active_optimizer = "default"
            else:
                raise ValueError("No valid optimizer configuration found in opt.yaml")

    gamma_scales_raw = sweep_config.get("gamma_scales", [])
    gamma_scales = [float(v) for v in gamma_scales_raw]
    if not gamma_scales:
        raise ValueError("gamma_sweep.yaml must define non-empty gamma_scales")

    R = int(sweep_config.get("R", exp_params.get("R", 100)))

    mode = runner_config.get("mode", "sequential")
    devices_config = runner_config.get("devices", "auto")

    if mode == "sequential":
        results = run_gamma_sweep_sequential(
            exp_params=exp_params,
            opt_params=opt_params,
            gamma_scales=gamma_scales,
            R=R,
        )
        meta: Dict[str, Any] = {
            "scenario": scenario,
            "optimizer": active_optimizer,
            "mode": "sequential",
            "method": "M-MMDVI",
            "gamma_scales": gamma_scales,
            "R": R,
            "experiment_params": exp_params,
            "optimizer_params": opt_params,
        }
    elif mode == "parallel":
        if isinstance(devices_config, str) and devices_config.lower() == "auto":
            devices = list(range(torch.cuda.device_count()))
        else:
            devices = [int(x) for x in str(devices_config).split(",") if x.strip()]

        if len(devices) < 2:
            raise RuntimeError(
                "Parallel gamma sweep requires >=2 GPUs, but only " f"{len(devices)} available."
            )

        if exp_params.get("seed", None) is not None:
            base_seed = int(exp_params.get("seed", 0))
        else:
            base_seed = 0

        results = run_gamma_sweep_parallel(
            exp_params=exp_params,
            opt_params=opt_params,
            gamma_scales=gamma_scales,
            R=R,
            devices=devices,
            base_seed=base_seed,
        )
        meta = {
            "scenario": scenario,
            "optimizer": active_optimizer,
            "mode": "parallel",
            "method": "M-MMDVI",
            "gamma_scales": gamma_scales,
            "R": R,
            "devices": devices,
            "experiment_params": exp_params,
            "optimizer_params": opt_params,
        }
    else:
        raise ValueError(f"Unknown runner mode: {mode}")

    results_dir: Path = save_gamma_sweep_results(
        results=results,
        meta=meta,
        results_root=results_root,
    )

    plot_gamma_sweep(
        gamma_scales=gamma_scales,
        mean_ed2=[r["mean_ED2"] for r in results],
        results_dir=results_dir,
    )

    print(f"Gamma sweep results saved to: {results_dir}")
    return results_dir
