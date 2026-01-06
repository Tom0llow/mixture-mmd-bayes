from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
import yaml  # type: ignore
from runners.run_parallel import run_parallel
from runners.run_sequantial import run_sequantial
from utils.output import save_experiment_results
from viz.plot_1d import plot_1d
from viz.plot_15d import plot_15d


def run(
    exp_config: Dict[str, Any],
    opt_config: Dict[str, Any],
    runner_config: Dict[str, Any],
    config_dir: Path,
    results_root: Path,
) -> None:
    """Run experiments described by configs and save results/plots.

    Args:
        exp_config: Experiment configuration mapping.
        opt_config: Optimizer configuration mapping.
        runner_config: Runner configuration mapping.
        config_dir: Path to configuration directory.
        results_root: Root path where results will be stored.
    """
    # Determine active scenarios
    raw_scenarios = exp_config.get("scenario", "default")
    if isinstance(raw_scenarios, str):
        active_scenarios = [raw_scenarios]
    elif isinstance(raw_scenarios, list):
        active_scenarios = raw_scenarios
    else:
        raise ValueError(
            "`scenario` must be a string or a list of strings in experiments.yaml"
        )

    for s in active_scenarios:
        if s not in exp_config:
            raise ValueError(f"Experiment scenario '{s}' not found in experiments.yaml")

    mode = runner_config["mode"]
    devices_config = runner_config["devices"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    parent_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parent_results_dir = results_root / parent_timestamp
    parent_results_dir.mkdir(parents=True, exist_ok=True)

    for active_scenario in active_scenarios:
        exp_params = exp_config[active_scenario]

        # Determine optimizer parameters for this scenario.
        if active_scenario in opt_config:
            opt_params = opt_config[active_scenario]
            active_optimizer = active_scenario
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
                    raise ValueError(
                        "No valid optimizer configuration found in opt.yaml"
                    )

        if mode == "sequential":
            print(f"Running SEQUENTIAL mode for scenario '{active_scenario}'...")
            results = run_sequantial(
                dim=exp_params.get("dim", 1),
                K=exp_params.get("K", 2),
                separation=exp_params.get("separation", 3.0),
                sigma=exp_params.get("sigma", 1.0),
                n_train=exp_params.get("n_train", 1000),
                n_test=exp_params.get("n_test", 2000),
                steps=opt_params.get("steps", 800),
                M=opt_params.get("M", 32),
                S=opt_params.get("S", 32),
                gamma_scale=opt_params.get("gamma_scale", 1e-4),
                lr=opt_params.get("lr", 5.0e-3),
                beta=opt_params.get("beta"),
                lambda_kl=opt_params.get("lambda_kl", 0.005),
                R=exp_params.get("R", 100),
                seed=exp_params.get("seed", None),
                device=device,
                dtype=torch.float64,
            )
        elif mode == "parallel":
            print(f"Running PARALLEL mode for scenario '{active_scenario}'...")
            if isinstance(devices_config, str) and devices_config.lower() == "auto":
                devices = (
                    list(range(torch.cuda.device_count()))
                    if torch.cuda.is_available()
                    else []
                )
            else:
                devices = [int(x) for x in str(devices_config).split(",") if x.strip()]

            if len(devices) < 2:
                raise RuntimeError(
                    "Parallel mode requires >=2 GPUs, but only "
                    f"{len(devices)} available."
                )

            results = run_parallel(
                devices=devices,
                dim=exp_params.get("dim", 1),
                K=exp_params.get("K", 2),
                separation=exp_params.get("separation", 3.0),
                sigma=exp_params.get("sigma", 1.0),
                n_train=exp_params.get("n_train", 1000),
                n_test=exp_params.get("n_test", 2000),
                steps=opt_params.get("steps", 800),
                M=opt_params.get("M", 32),
                S=opt_params.get("S", 32),
                gamma_scale=opt_params.get("gamma_scale", 1e-4),
                lr=opt_params.get("lr", 5.0e-3),
                beta=opt_params.get("beta"),
                lambda_kl=opt_params.get("lambda_kl", 0.005),
                R=exp_params.get("R", 100),
                base_seed=exp_params.get("seed", 0)
                if exp_params.get("seed", None) is not None
                else 0,
            )
        else:
            raise ValueError(f"Unknown runner mode: {mode}")

        # Print results and save
        print("\n" + "=" * 50)
        print(f"CONFIGURATION (scenario: {active_scenario})")
        print("=" * 50)
        print(yaml.safe_dump(exp_params, default_flow_style=False))
        print(f"Optimizer: {active_optimizer}")
        print(yaml.safe_dump(opt_params, default_flow_style=False))
        print("Runner:")
        print(yaml.safe_dump(runner_config, default_flow_style=False))

        print("\n" + "=" * 50)
        print("RESULTS")
        print("=" * 50)
        for method, stats in results.items():
            mean_ed2 = stats["mean_ED2"]
            se_ed2 = stats["stderr"]
            runs = stats["runs"]
            print(f"{method}: Mean ED^2 = {mean_ed2:.4f} ± {se_ed2:.4f} (runs={runs})")

        results_dir = save_experiment_results(
            results=results,
            exp_params=exp_params,
            opt_params=opt_params,
            runner_config=runner_config,
            active_scenario=active_scenario,
            active_optimizer=active_optimizer,
            config_dir=config_dir,
            results_root=results_root,
            parent_timestamp=parent_timestamp,
        )
        print(f"\nResults for scenario '{active_scenario}' saved to: {results_dir}")

        # Plot once per scenario
        try:
            save_dir = str(results_dir)
            dim = exp_params.get("dim", 1)
            if dim == 1:
                print("Generating 1D plots...\n")
                plot_1d(
                    dim=exp_params.get("dim", 1),
                    K=exp_params.get("K", 2),
                    separation=exp_params.get("separation", 3.0),
                    sigma=exp_params.get("sigma", 1.0),
                    n_train=exp_params.get("n_train", 800),
                    n_test=exp_params.get("n_test", 2000),
                    steps=opt_params.get("steps", 1200),
                    M=opt_params.get("M", 32),
                    S=opt_params.get("S", 32),
                    gamma_scale=opt_params.get("gamma_scale", 1e-3),
                    seed=exp_params.get("seed", 0),
                    device=device,
                    dtype=torch.float64,
                    show=False,
                    save_dir=save_dir,
                    mode=mode,
                    devices=devices_config,
                )
            elif dim == 15:
                print("Generating 15D plots (PCA)...\n")
                plot_15d(
                    K=exp_params.get("K", 4),
                    separation=exp_params.get("separation", 3.0),
                    sigma=exp_params.get("sigma", 1.0),
                    n_train=exp_params.get("n_train", 2000),
                    n_test=exp_params.get("n_test", 5000),
                    steps=opt_params.get("steps", 1200),
                    M=opt_params.get("M", 64),
                    S=opt_params.get("S", 64),
                    gamma_scale=opt_params.get("gamma_scale", 1e-3),
                    seed=exp_params.get("seed", 1),
                    device=device,
                    dtype=torch.float64,
                    show=False,
                    save_dir=save_dir,
                    mode=mode,
                    devices=devices_config,
                )
            else:
                print(f"No plotting implemented for dim={dim}; skipping plots.")
        except Exception as e:
            print(f"Plotting failed for scenario '{active_scenario}': {e}")
