from pathlib import Path

import torch
import yaml  # type: ignore
from runners.run_parallel import run_parallel
from runners.run_sequantial import run_sequantial
from utils.config import load_config
from utils.output import save_experiment_results

if __name__ == "__main__":
    # Load configuration directory
    config_dir = Path(__file__).parent.parent / "configs"
    results_root = Path(__file__).parent.parent / "results"

    # Load configuration files
    exp_config = load_config(config_dir / "experiments.yaml")
    opt_config = load_config(config_dir / "opt.yaml")
    runner_config = load_config(config_dir / "runner.yaml")

    # Get experiment scenario(s). Accept a string or a list of names.
    raw_scenarios = exp_config.get("scenario", "default")
    if isinstance(raw_scenarios, str):
        active_scenarios = [raw_scenarios]
    elif isinstance(raw_scenarios, list):
        active_scenarios = raw_scenarios
    else:
        raise ValueError(
            "`scenario` must be a string or a list of strings in experiments.yaml"
        )

    # Validate scenarios exist
    for s in active_scenarios:
        if s not in exp_config:
            raise ValueError(f"Experiment scenario '{s}' not found in experiments.yaml")

    # Note: optimizer selection is handled per-scenario below. The global default
    # is read from opt.yaml via `opt_config.get('optimizer')` when a scenario
    # does not specify its own optimizer.

    # Get runner configuration
    mode = runner_config["mode"]
    devices_config = runner_config["devices"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a single parent timestamp directory for all scenarios in this run.
    # All scenario results will be saved as subdirectories under this timestamp.
    from datetime import datetime

    parent_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parent_results_dir = results_root / parent_timestamp
    parent_results_dir.mkdir(parents=True, exist_ok=True)

    # Run each requested scenario sequentially in this process.
    for active_scenario in active_scenarios:
        exp_params = exp_config[active_scenario]

        # Determine optimizer parameters for this scenario.
        # Priority:
        # 1) If opt.yaml contains a key matching the scenario name, use it.
        # 2) Else, if scenario specifies optimizer in exp_params, use that mapping.
        # 3) Else, fall back to global default in opt.yaml (opt_config['optimizer']).
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
                seed=exp_params.get("seed", None),
                device=device,
                dtype=torch.float64,
            )
        elif mode == "parallel":
            print(f"Running PARALLEL mode for scenario '{active_scenario}'...")
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
                    f"Parallel mode requires >=2 GPUs, "
                    f"but only {len(devices)} available."
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
                base_seed=exp_params.get("seed", 0)
                if exp_params.get("seed", None) is not None
                else 0,
            )

        # Print results for this scenario
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

        # Save results to scenario subdirectory under parent timestamp directory
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
        print(f"\nResults for scenario '{active_scenario}' saved to: {results_dir}\n")
    # end for scenarios
    # end main
