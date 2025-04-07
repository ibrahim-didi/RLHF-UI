"""
Visualization package for RLHF UI.
"""

from .logger import (
    init_wandb,
    log_metrics,
    log_image_comparison,
    log_reward_predictions,
    log_model,
    finish_run
)

from .dashboard import (
    get_wandb_url,
    get_wandb_api_key,
    create_embedded_iframe,
    generate_live_metrics_chart,
    generate_separate_metric_charts
)

__all__ = [
    # Logger functions
    "init_wandb",
    "log_metrics",
    "log_image_comparison",
    "log_reward_predictions",
    "log_model",
    "finish_run",
    
    # Dashboard functions
    "get_wandb_url",
    "get_wandb_api_key",
    "create_embedded_iframe",
    "generate_live_metrics_chart",
    "generate_separate_metric_charts"
] 