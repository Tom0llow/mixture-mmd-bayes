from pathlib import Path

from runners.run import run
from utils.config import load_config

if __name__ == "__main__":
    config_dir = Path(__file__).parent.parent / "configs"
    results_root = Path(__file__).parent.parent / "results"

    exp_config = load_config(config_dir / "experiments.yaml")
    opt_config = load_config(config_dir / "opt.yaml")
    runner_config = load_config(config_dir / "runner.yaml")

    run(
        exp_config=exp_config,
        opt_config=opt_config,
        runner_config=runner_config,
        config_dir=config_dir,
        results_root=results_root,
    )
