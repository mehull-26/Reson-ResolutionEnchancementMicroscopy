"""Configuration loader for Reson pipeline."""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_path : str
        Path to the configuration YAML file.

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required sections
    required_sections = ['version', 'enhancement']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section '{section}' in config")

    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure and parameters.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary to validate.

    Returns
    -------
    bool
        True if valid, raises ValueError otherwise.
    """
    # Check version
    if config.get('version') != 'v0':
        raise ValueError(
            f"Config version mismatch. Expected 'v0', got '{config.get('version')}'")

    # Check enhancement section
    enhancement = config.get('enhancement', {})
    if 'modules' not in enhancement:
        raise ValueError("Enhancement section must contain 'modules'")

    return True
