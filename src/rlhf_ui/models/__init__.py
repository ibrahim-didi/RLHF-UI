"""
Models module for RLHF UI.

This module contains model definitions, trainers, and utilities for working with
neural networks in the RLHF workflow.
"""

from .trainer import RewardModel, RewardModelTrainer
from .embedding import ImageEmbeddingModel

__all__ = ["RewardModel", "RewardModelTrainer", "ImageEmbeddingModel"]
