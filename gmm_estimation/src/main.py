from pathlib import Path

import torch
import yaml  # type: ignore
from runners.run_parallel import run_parallel
from runners.run_sequantial import run_sequantial
from utils.config import load_config

if __name__ == "__main__":
    # Load configuration directory
    config_dir = Path(__file__).parent.parent / "configs"

    # Load configuration files
    exp_config = load_config(config_dir / "experiments.yaml")
    opt_config = load_config(config_dir / "opt.yaml")
    runner_config = load_config(config_dir / "runner.yaml")

    # Get default experiment scenario parameters
    active_scenario = exp_config.get("scenario", "default")
    if active_scenario not in exp_config:
        raise ValueError(
            f"Experiment scenario '{active_scenario}' not found in experiments.yaml"
        )
    exp_params = exp_config[active_scenario]

    # Get default optimizer parameters
    active_optimizer = opt_config.get("optimizer", "default")
    if active_optimizer not in opt_config:
        raise ValueError(f"Optimizer '{active_optimizer}' not found in opt.yaml")
    opt_params = opt_config[active_optimizer]

    # Get runner configuration
    mode = runner_config["mode"]
    devices_config = runner_config["devices"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if mode == "sequential":
        print("Running SEQUENTIAL mode...")
        results = run_sequantial(
            dim=exp_params["dim"],
            K=exp_params["K"],
            separation=exp_params["separation"],
            sigma=exp_params["sigma"],
            n_train=exp_params["n_train"],
            n_test=exp_params["n_test"],
            steps=opt_params["steps"],
            M=opt_params["M"],
            S=opt_params["S"],
            gamma_scale=opt_params["gamma_scale"],
            R=exp_params["R"],
            seed=exp_params["seed"],
            device=device,
            dtype=torch.float64,
        )
    elif mode == "parallel":
        print("Running PARALLEL mode...")
        if devices_config.lower() == "auto":
            devices = (
                list(range(torch.cuda.device_count()))
                if torch.cuda.is_available()
                else []
            )
        else:
            devices = [int(x) for x in devices_config.split(",") if x.strip()]

        if len(devices) < 2:
            raise RuntimeError(
                f"Parallel mode requires >=2 GPUs, but only {len(devices)} available."
            )

        results = run_parallel(
            devices=devices,
            dim=exp_params["dim"],
            K=exp_params["K"],
            separation=exp_params["separation"],
            sigma=exp_params["sigma"],
            n_train=exp_params["n_train"],
            n_test=exp_params["n_test"],
            steps=opt_params["steps"],
            M=opt_params["M"],
            S=opt_params["S"],
            gamma_scale=opt_params["gamma_scale"],
            R=exp_params["R"],
            base_seed=exp_params["seed"] if exp_params["seed"] is not None else 0,
        )

    # Print results
    # Print configuration (experiment, optimizer, runner)
    print("\n" + "=" * 50)
    print("CONFIGURATION")
    print("=" * 50)
    print(f"Scenario: {active_scenario}")
    print(yaml.safe_dump(exp_params, default_flow_style=False))
    print(f"Optimizer: {active_optimizer}")
    print(yaml.safe_dump(opt_params, default_flow_style=False))
    print("Runner:")
    print(yaml.safe_dump(runner_config, default_flow_style=False))

    # Print results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    for method, stats in results.items():
        mean_ed2 = stats["mean_ED2"]
        se_ed2 = stats["stderr"]
        runs = stats["runs"]
        print(f"{method}: Mean ED^2 = {mean_ed2:.4f} ± {se_ed2:.4f} (runs={runs})")
