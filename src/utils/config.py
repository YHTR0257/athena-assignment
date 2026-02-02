import json
import tomllib
from pathlib import Path
from typing import Union

def load_config(config_path: Union[str, Path]) -> dict:
    """Load configuration file (supports .toml and .json).

    Args:
        config_path: Path to config file (.toml or .json)

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If file extension is not supported
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".toml":
        with open(path, "rb") as f:
            return tomllib.load(f)
    elif suffix == ".json":
        with open(path, "r") as f:
            return json.load(f)
    else:
        raise ValueError(
            f"Unsupported config file extension: {suffix}\n"
            f"Supported extensions: .toml, .json"
        )
