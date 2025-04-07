"""
UI components for the RLHF UI application.

This module contains Gradio-based UI components and tabs for the application.
"""

from .tabs import (
    PreferenceCollectionTab,
    RewardModelTab,
    InferenceTab,
    RLHFOptimizationTab
)

__all__ = [
    "PreferenceCollectionTab",
    "RewardModelTab", 
    "InferenceTab", 
    "RLHFOptimizationTab"
] 