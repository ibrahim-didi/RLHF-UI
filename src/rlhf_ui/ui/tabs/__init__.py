"""
Tab components for the RLHF UI application.

This module contains individual tab implementations for the Gradio interface.
"""

from .preference_collection import PreferenceCollectionTab
from .reward_model import RewardModelTab
from .inference import InferenceTab
from .rlhf_optimization import RLHFOptimizationTab

__all__ = [
    "PreferenceCollectionTab",
    "RewardModelTab",
    "InferenceTab",
    "RLHFOptimizationTab"
] 