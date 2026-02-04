import json
import tomllib
import yaml
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
    elif suffix == ".yaml" or suffix == ".yml":
        with open(path, "r") as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(
            f"Unsupported config file extension: {suffix}\n"
            f"Supported extensions: .toml, .json, .yaml, .yml"
        )

def load_config_section(config_path: Union[str, Path], section: str) -> dict:
    """Load a specific section from the configuration file.

    Args:
        config_path: Path to config file (.toml or .json)
        section: Section name to extract
    Returns:
        Configuration section as a dictionary
    """
    config = load_config(config_path)
    return config.get(section, {})
