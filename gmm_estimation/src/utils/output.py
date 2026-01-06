import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def save_experiment_results(
    results: Dict[str, Dict[str, float]],
    exp_params: Dict[str, Any],
    opt_params: Dict[str, Any],
    runner_config: Dict[str, Any],
    active_scenario: str,
    active_optimizer: str,
    config_dir: Path,
    results_root: Path,
    parent_timestamp: str | None = None,
) -> Path:
    """Save experiment results and configuration to timestamped directory.

    Args:
        results: Experiment results from runner
        exp_params: Experiment scenario parameters
        opt_params: Optimizer parameters
        runner_config: Runner configuration
        active_scenario: Name of active scenario
        active_optimizer: Name of active optimizer configuration
        config_dir: Path to configs directory (unused, kept for compatibility)
        results_root: Path to results root directory
        parent_timestamp: Optional timestamp string for parent directory.
                         If provided, uses existing parent directory.
                         If None, creates a new timestamped parent directory.

    Returns:
        Path to the created results directory
    """
    # Ensure results root directory exists
    results_root.mkdir(parents=True, exist_ok=True)

    # Use provided parent_timestamp or create a new one
    if parent_timestamp is None:
        parent_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    parent_dir = results_root / parent_timestamp
    parent_dir.mkdir(parents=True, exist_ok=True)

    # Make a filesystem-safe scenario directory name and create it under the
    # parent timestamp directory.
    safe_scenario = str(active_scenario).replace(" ", "_")
    results_dir = parent_dir / safe_scenario
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save results to JSON
    results_file = results_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Save metadata (contains all experiment configuration)
    meta = {
        "timestamp": parent_timestamp,
        "scenario": active_scenario,
        "optimizer": active_optimizer,
        "mode": runner_config.get("mode", "unknown"),
        "experiment_params": exp_params,
        "optimizer_params": opt_params,
    }
    meta_file = results_dir / "meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    return results_dir
