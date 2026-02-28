import argparse
from pathlib import Path

from gamma_sweep.gamma_sweep import run_gamma_sweep
from runners.run import run
from utils.config import load_config


def main() -> None:
    """Parse CLI arguments and run either experiments or gamma sweep."""
    parser = argparse.ArgumentParser(description="Run standard experiments or gamma sweep.")
    parser.add_argument(
        "--mode",
        choices=["experiment", "gamma_sweep"],
        default="experiment",
        help="Select runner mode.",
    )
    args = parser.parse_args()

    config_dir = Path(__file__).parent.parent / "configs"
    results_root = Path(__file__).parent.parent / "results"

    exp_config = load_config(config_dir / "experiments.yaml")
    opt_config = load_config(config_dir / "opt.yaml")
    runner_config = load_config(config_dir / "runner.yaml")

    if args.mode == "experiment":
        run(
            exp_config=exp_config,
            opt_config=opt_config,
            runner_config=runner_config,
            config_dir=config_dir,
            results_root=results_root,
        )
    elif args.mode == "gamma_sweep":
        sweep_config = load_config(config_dir / "gamma_sweep.yaml")
        run_gamma_sweep(
            sweep_config=sweep_config,
            exp_config=exp_config,
            opt_config=opt_config,
            runner_config=runner_config,
            results_root=results_root,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
