"""
Dashboard integration for real-time visualization.

This module provides functions to integrate with Weights & Biases
for real-time dashboard visualization within the Gradio UI.
"""

import logging
import time
from typing import Optional, Dict, Any, List
import os
from pathlib import Path
import json

import wandb
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

logger = logging.getLogger(__name__)

def get_wandb_url() -> Optional[str]:
    """
    Get the URL for the current wandb run.
    
    Returns:
        str: URL of the current wandb run or None if not available
    """
    if wandb.run is None:
        return None
    
    try:
        return wandb.run.get_url()
    except Exception as e:
        logger.error(f"Error getting wandb URL: {e}")
        return None

def get_wandb_api_key() -> Optional[str]:
    """
    Get the wandb API key from environment or wandb directory.
    
    Returns:
        str: wandb API key or None if not found
    """
    # First check environment
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        return api_key
    
    # Then check the wandb config file
    try:
        home_dir = Path.home()
        wandb_dir = home_dir / ".wandb"
        
        if not wandb_dir.exists():
            return None
        
        config_file = wandb_dir / "settings"
        if not config_file.exists():
            return None
            
        with open(config_file) as f:
            settings = json.load(f)
            return settings.get("api_key")
    except Exception as e:
        logger.error(f"Error reading wandb config: {e}")
        return None
    
    return None

def create_embedded_iframe(wandb_url: str) -> str:
    """
    Create an HTML iframe for embedding a wandb dashboard.
    
    Args:
        wandb_url: URL of the wandb dashboard
        
    Returns:
        str: HTML snippet with iframe
    """
    if not wandb_url:
        return """
        <div style="height: 600px; display: flex; justify-content: center; align-items: center; border: 1px solid #ddd; border-radius: 4px;">
            <p style="color: #666;">Weights & Biases dashboard not available</p>
        </div>
        """
    
    # Extract run ID and project from URL
    # URL format: https://wandb.ai/username/project/runs/run_id
    try:
        parts = wandb_url.strip("/").split("/")
        if "runs" in parts:
            run_idx = parts.index("runs")
            if run_idx + 1 < len(parts):
                run_id = parts[run_idx + 1]
                username = parts[2]  # After wandb.ai/
                project = parts[3]   # After username
                
                # Direct embed URL for the run
                embed_url = f"https://wandb.ai/embed/run/{username}/{project}/{run_id}"
                
                return f"""
                <div style="height: 600px; border: 1px solid #ddd; border-radius: 4px; overflow: hidden;">
                    <iframe 
                        src="{embed_url}" 
                        style="width: 100%; height: 100%; border: none;"
                        allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
                        allowfullscreen>
                    </iframe>
                </div>
                """
    except Exception as e:
        logger.error(f"Error creating iframe from URL {wandb_url}: {e}")
    
    # Fallback to direct URL
    return f"""
    <div style="height: 600px; border: 1px solid #ddd; border-radius: 4px; overflow: hidden;">
        <iframe 
            src="{wandb_url}" 
            style="width: 100%; height: 100%; border: none;"
            allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen>
        </iframe>
    </div>
    """

def generate_live_metrics_chart(
    run_id: Optional[str] = None,
    metric_keys: Optional[List[str]] = None
) -> Optional[go.Figure]:
    """
    Generate a plotly chart with live metrics from wandb.
    Creates individual subplots for each metric type.
    
    Args:
        run_id: Optional run ID to fetch metrics from (current run if None)
        metric_keys: List of metrics to include (all metrics if None)
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with metrics
    """
    if wandb.run is None and run_id is None:
        return None
    
    try:
        # Get the API
        api = wandb.Api()
        
        # Use current run if no run_id provided
        if run_id is None and wandb.run is not None:
            run_id = wandb.run.id
            
        # Get username and project from current run
        if wandb.run is not None:
            username = wandb.run.entity
            project = wandb.run.project
        else:
            # Default to current settings
            username = os.environ.get("WANDB_ENTITY", "")
            project = os.environ.get("WANDB_PROJECT", "rlhf-reward-model")
        
        # Get run
        run = api.run(f"{username}/{project}/{run_id}")
        
        # Get history with all metrics
        history = run.history()
        
        if history.empty:
            logger.warning("No history data available in W&B run")
            return None
            
        # Log available columns for debugging
        logger.info(f"Available metrics in W&B history: {list(history.columns)}")
        
        # Define metric groups with colors
        metrics_config = {
            'Accuracy': {
                'metrics': ['train/accuracy', 'train/running_accuracy', 'train/best_accuracy'],
                'colors': ['rgb(31, 119, 180)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)']
            },
            'Loss': {
                'metrics': ['train/loss', 'train/running_loss', 'train/batch_loss'],
                'colors': ['rgb(255, 127, 14)', 'rgb(148, 103, 189)', 'rgb(140, 86, 75)']
            }
        }
        
        # Get x-axis values (prefer _step if available, otherwise use index)
        if '_step' in history.columns:
            x_values = history['_step']
        else:
            x_values = history.index

        # Create subplots - one for each metric group
        num_groups = len(metrics_config)
        fig = go.Figure()
        
        # Track the metrics we've added for layout
        added_metrics = []
        
        # Create traces for each metric group
        for group_name, config in metrics_config.items():
            available_metrics = [m for m in config['metrics'] if m in history.columns]
            
            if not available_metrics:
                continue
                
            for i, metric in enumerate(available_metrics):
                try:
                    y_values = history[metric].ffill().bfill()
                    if y_values.isnull().all():
                        logger.warning(f"Metric {metric} contains only null values after fill. Skipping.")
                        continue
                    
                    color = config['colors'][i % len(config['colors'])]
                    logger.debug(f"Adding trace for {group_name} metric: {metric}")
                    
                    # Add trace
                    fig.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=y_values,
                            name=metric.replace('train/', ''),
                            mode='lines+markers',
                            line=dict(width=2, color=color),
                            marker=dict(size=4),
                            xaxis='x',
                            yaxis=f'y{len(added_metrics) + 1}'
                        )
                    )
                    added_metrics.append(metric)
                    
                except Exception as e:
                    logger.error(f"Error adding trace for {group_name} metric {metric}: {e}")
                    logger.exception("Detailed traceback for trace error:")

        if not added_metrics:
            logger.warning("No metrics were successfully added to the plot")
            return None

        # Update layout to create separate subplots
        layout_updates = {}
        subplot_height = 300  # Height per subplot
        
        # Calculate total figure height based on number of metrics
        total_height = subplot_height * len(added_metrics)
        
        # Update layout with grid
        layout_updates.update({
            'height': total_height,
            'showlegend': True,
            'grid': {
                'rows': len(added_metrics),
                'columns': 1,
                'pattern': 'independent'
            }
        })
        
        # Add individual axis configurations
        for i, metric in enumerate(added_metrics, 1):
            # X-axis (only show label for bottom plot)
            layout_updates[f'xaxis{i}'] = {
                'title': 'Step' if i == len(added_metrics) else None,
                'showgrid': True,
                'domain': [0, 1],
                'anchor': f'y{i}'
            }
            
            # Y-axis
            layout_updates[f'yaxis{i}'] = {
                'title': metric.replace('train/', '').title(),
                'showgrid': True,
                'domain': [1 - (i / len(added_metrics)), 1 - ((i - 1) / len(added_metrics))],
                'anchor': f'x{i}'
            }

        # Update figure layout
        fig.update_layout(
            **layout_updates,
            margin=dict(l=50, r=50, t=30, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=10)
        )

        return fig
        
    except Exception as e:
        logger.error(f"Error generating metrics chart: {e}")
        logger.exception("Detailed traceback for chart generation error:")
        return None

def generate_separate_metric_charts(
    run_id: Optional[str] = None,
    metric_keys: Optional[List[str]] = None
) -> List[Optional[go.Figure]]:
    """
    Generate separate plotly charts for different metric groups from wandb.
    
    Args:
        run_id: Optional run ID to fetch metrics from (current run if None)
        metric_keys: List of metrics to include (all metrics if None)
        
    Returns:
        List[plotly.graph_objects.Figure]: List of plotly figures, one for each metric group
    """
    if wandb.run is None and run_id is None:
        return []
    
    try:
        # Get the API
        api = wandb.Api()
        
        # Use current run if no run_id provided
        if run_id is None and wandb.run is not None:
            run_id = wandb.run.id
            
        # Get username and project from current run
        if wandb.run is not None:
            username = wandb.run.entity
            project = wandb.run.project
        else:
            # Default to current settings
            username = os.environ.get("WANDB_ENTITY", "")
            project = os.environ.get("WANDB_PROJECT", "rlhf-reward-model")
        
        # Get run
        run = api.run(f"{username}/{project}/{run_id}")
        
        # Get history with all metrics
        history = run.history()
        
        if history.empty:
            logger.warning("No history data available in W&B run")
            return []
            
        # Log available columns for debugging
        logger.info(f"Available metrics in W&B history: {list(history.columns)}")
        
        # Define metric groups with colors
        metrics_config = {
            'Accuracy': {
                'metrics': ['train/accuracy', 'train/running_accuracy', 'train/best_accuracy'],
                'colors': ['rgb(31, 119, 180)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)']
            },
            'Loss': {
                'metrics': ['train/loss', 'train/running_loss', 'train/batch_loss'],
                'colors': ['rgb(255, 127, 14)', 'rgb(148, 103, 189)', 'rgb(140, 86, 75)']
            }
        }
        
        # Get x-axis values (prefer _step if available, otherwise use index)
        if '_step' in history.columns:
            x_values = history['_step']
        else:
            x_values = history.index

        figures = []
        
        # Create a separate figure for each metric group
        for group_name, config in metrics_config.items():
            available_metrics = [m for m in config['metrics'] if m in history.columns]
            
            if not available_metrics:
                continue
            
            # Create a new figure for this metric group
            group_fig = go.Figure()
            metrics_added = False
                
            for i, metric in enumerate(available_metrics):
                try:
                    y_values = history[metric].ffill().bfill()
                    if y_values.isnull().all():
                        logger.warning(f"Metric {metric} contains only null values after fill. Skipping.")
                        continue
                    
                    color = config['colors'][i % len(config['colors'])]
                    logger.debug(f"Adding trace for {group_name} metric: {metric}")
                    
                    # Create descriptive name from metric
                    display_name = metric.replace('train/', '').replace('_', ' ').title()
                    
                    # Add trace to this group's figure
                    group_fig.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=y_values,
                            name=display_name,
                            mode='lines+markers',
                            line=dict(width=2.5, color=color),
                            marker=dict(size=5)
                        )
                    )
                    metrics_added = True
                    
                except Exception as e:
                    logger.error(f"Error adding trace for {group_name} metric {metric}: {e}")
                    logger.exception("Detailed traceback for trace error:")

            if metrics_added:
                # Update this figure's layout with consistent styling
                group_fig.update_layout(
                    title=f"{group_name} Metrics",
                    height=320,  # Fixed height for consistent appearance in cards
                    showlegend=True,
                    xaxis_title="Training Step",
                    yaxis_title=group_name,
                    margin=dict(l=40, r=40, t=50, b=40),  # Balanced margins
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family="Arial, sans-serif", size=12),
                    title_x=0.5,  # Center the title
                    title_font=dict(size=16),  # Slightly larger title
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5,
                        font=dict(size=11)
                    ),
                    # Add grid for better readability
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(200,200,200,0.2)',
                        tickfont=dict(size=11)
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(200,200,200,0.2)',
                        tickfont=dict(size=11)
                    )
                )
                figures.append(group_fig)

        return figures
        
    except Exception as e:
        logger.error(f"Error generating separate metric charts: {e}")
        logger.exception("Detailed traceback for chart generation error:")
        return [] 