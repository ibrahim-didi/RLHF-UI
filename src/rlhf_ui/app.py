# src/rlhf_ui/app.py
"""
Main application file for the RLHF UI using Gradio.
"""

import os
import logging
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import webbrowser
from datetime import datetime
import sys

import gradio as gr
import numpy as np
from PIL import Image
import torch
import urllib.parse
import wandb

from rlhf_ui.config import AppConfig, load_config
from rlhf_ui.data.storage import PreferenceDataStorage
from rlhf_ui.data.sampler import ImagePairSampler
from rlhf_ui.models.trainer import RewardModelTrainer
from rlhf_ui.models.embedding import ImageEmbeddingModel
from rlhf_ui.visualization import init_wandb, finish_run
from rlhf_ui.ui.tabs import PreferenceCollectionTab, RewardModelTab, InferenceTab, RLHFOptimizationTab
from rlhf_ui.ui.tabs.base import TabRegistry

# Get logger for this module
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()

class RLHFWebUI:
    """
    Gradio-based Web UI for RLHF workflows.
    """
    
    def __init__(self, config: AppConfig):
        """
        Initialize the RLHF Web UI.
        
        Args:
            config: Application configuration
        """
        self.config = config
        
        # Ensure directories exist
        self.config.image_folder.mkdir(exist_ok=True, parents=True)
        self.config.output_folder.mkdir(exist_ok=True, parents=True)
        self.config.model_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize tab registry
        self.tab_registry = TabRegistry()
        
        # Initialize tab components
        self._init_tabs()
        
        # Gradio app instance
        self.app: gr.Blocks = None
        
        # Build Gradio interface
        self.build_interface()
    
    def _init_tabs(self) -> None:
        """Initialize tab components."""
        # Create each tab
        preference_tab = PreferenceCollectionTab(self.config)
        reward_model_tab = RewardModelTab(self.config)
        inference_tab = InferenceTab(self.config)
        rlhf_tab = RLHFOptimizationTab(self.config)
        
        # Register tabs
        self.tab_registry.register("preference_collection", preference_tab)
        self.tab_registry.register("reward_model", reward_model_tab)
        self.tab_registry.register("inference", inference_tab)
        self.tab_registry.register("rlhf_optimization", rlhf_tab)
    
    def build_interface(self) -> None:
        """Build the Gradio interface with all tabs."""
        # Create the app with custom theme
        self.app = gr.Blocks(
            title="RLHF UI",
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="indigo",
            ),
            css="""
            .feedback-button { min-height: 60px; font-size: 16px; }
            .gallery-image { min-height: 400px; }
            
            /* Card-like styling for metric plots */
            .metric-box {
                margin-bottom: 24px;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                padding: 16px;
                background-color: white;
                border: 1px solid rgba(0, 0, 0, 0.05);
                transition: transform 0.2s, box-shadow 0.2s;
                overflow: hidden;
            }
            
            .metric-box:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
            }
            
            .metric-header {
                margin-top: 0 !important;
                margin-bottom: 12px !important;
                padding-bottom: 8px;
                border-bottom: 1px solid rgba(0, 0, 0, 0.05);
            }
            
            .metric-box h4 {
                margin-top: 0;
                margin-bottom: 10px;
                color: #444;
                font-weight: 600;
            }
            
            /* Make plots have adequate height */
            .plot-container {
                min-height: 350px;
                margin-top: 10px;
            }
            """
        )
        
        # Build UI components
        with self.app:
            gr.Markdown("# Reinforcement Learning from Human Feedback (RLHF) UI")
            
            with gr.Tabs() as tabs:
                # Build each tab
                for tab_id, tab in self.tab_registry.get_all().items():
                    with gr.Tab(tab.get_name()):
                        tab.build(self)
                        
            # Register event handlers
            for tab in self.tab_registry.get_all().values():
                tab.register_event_handlers()
    
    def launch(self, share: bool = False, debug: bool = False) -> None:
        """
        Launch the Gradio web interface.
        
        Args:
            share: Whether to share the app publicly
            debug: Whether to enable debug mode
        """
        self.app.launch(share=share, debug=debug)

def main():
    """Main entry point for the application."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create and launch the web UI
    ui = RLHFWebUI(config)
    ui.launch(debug=True)

if __name__ == "__main__":
    main()