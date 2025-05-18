# utils/config.py
import os
import json
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration file
    """
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
