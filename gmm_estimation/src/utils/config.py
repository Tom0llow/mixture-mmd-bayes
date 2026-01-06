from pathlib import Path
from typing import Any

import yaml  # type: ignore


def load_config(config_path: Path) -> Any:
    """Load YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
