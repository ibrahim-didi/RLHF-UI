# src/rlhf_ui/config.py
"""
Configuration settings for the RLHF UI application.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class AppConfig:
    """Application configuration."""
    # Data paths
    image_folder: Path = Path("./images")
    output_folder: Path = Path("./preference_data")
    model_output_dir: Path = Path("./reward_model")
    
    # UI settings
    window_size: tuple = (1400, 800)
    theme: str = "light"  # Options: "light", "dark"
    
    # Sampling strategy settings
    sampling_strategy: str = "active"  # Options: "random", "active", "diversity"
    
    # Training settings
    training_epochs: int = 20
    learning_rate: float = 1e-4
    batch_size: int = 8
    use_text_embeddings: bool = False
    
    # Additional settings
    use_gpu: bool = True
    data_augmentation: bool = True
    auto_backup: bool = True
    backup_interval: int = 10  # Minutes
    
    # Custom params (for extensibility)
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                result[key] = str(value)
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppConfig':
        """Create config from dictionary."""
        config = cls()
        for key, value in data.items():
            if key in ["image_folder", "output_folder", "model_output_dir"] and value is not None:
                setattr(config, key, Path(value))
            elif hasattr(config, key):
                setattr(config, key, value)
            else:
                config.custom_params[key] = value
        return config

def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """
    Load configuration from a file or use defaults.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        AppConfig object
    """
    if config_path is None:
        # Check for config in common locations
        locations = [
            Path("./config.json"),
            Path.home() / ".config" / "rlhf_ui" / "config.json",
            Path("/etc/rlhf_ui/config.json")
        ]
        
        for loc in locations:
            if loc.exists():
                config_path = loc
                break
    
    if config_path and config_path.exists():
        with open(config_path, "r") as f:
            config_data = json.load(f)
        return AppConfig.from_dict(config_data)
    
    # No config found, use defaults
    return AppConfig()

def save_config(config: AppConfig, config_path: Path) -> None:
    """
    Save configuration to a file.
    
    Args:
        config: AppConfig object
        config_path: Path to save the configuration
    """
    # Create parent directories if they don't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)