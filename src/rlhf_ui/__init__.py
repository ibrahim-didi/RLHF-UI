"""
RLHF UI - A graphical interface for RLHF workflows.

This package provides tools for collecting human preferences,
training reward models, and performing RLHF optimization.
"""

__version__ = "0.1.0"
__author__ = "Ibrahim Didi"

from rlhf_ui.config import AppConfig, load_config
from rlhf_ui.app import RLHFWebUI

__all__ = ["RLHFWebUI", "AppConfig", "load_config"]
