"""
RLHF (Reinforcement Learning from Human Feedback) optimization module.

This module provides tools for optimizing generative models using reward models 
trained on human preferences.
"""

from .optimizer import RLHFOptimizer

__all__ = ["RLHFOptimizer"]
