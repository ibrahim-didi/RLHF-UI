"""
Inference tab for the RLHF UI application.

This module contains the implementation of the Inference tab, which allows
users to run inference with trained reward models.
"""

from typing import Any, Dict, Optional, List, Tuple
import os
import logging
import json
import random
from pathlib import Path

import gradio as gr
import torch
import numpy as np
from PIL import Image
from matplotlib.figure import Figure

from rlhf_ui.config import AppConfig
from rlhf_ui.models.embedding import ImageEmbeddingModel
from rlhf_ui.models.trainer import RewardModelTrainer
from rlhf_ui.ui.tabs.base import BaseTab

# Get logger for this module
logger = logging.getLogger(__name__)

class InferenceTab(BaseTab):
    """
    Tab for running inference with trained reward models.
    
    This tab allows users to:
    - Run inference on single images
    - Perform batch analysis on image folders
    - Visualize reward distributions
    """
    
    def __init__(self, config: AppConfig):
        """
        Initialize the inference tab.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.embedding_model = ImageEmbeddingModel()
        self.trainer = self._init_trainer()
        
        # UI components (will be set during build)
        # Single image inference
        self.input_image = None
        self.prompt_input = None
        self.prediction_output = None
        self.predict_btn = None
        
        # Batch inference
        self.input_folder = None
        self.sample_size = None
        self.batch_predict_btn = None
        self.batch_results_plot = None
        self.batch_results_table = None
        
        # Model selection
        self.model_dropdown = None
        self.refresh_models_btn = None
        self.load_model_btn = None
        self.model_info_text = None
        
        # Current loaded model
        self.current_model = None
        self.model_loaded = False
    
    def _init_trainer(self) -> Optional[RewardModelTrainer]:
        """Initialize the reward model trainer."""
        try:
            preference_file = self.config.output_folder / "preferences.csv"
            trainer = RewardModelTrainer(
                preference_file=preference_file,
                image_folder=self.config.image_folder,
                model_output_dir=self.config.model_output_dir
            )
            logger.info("Reward model trainer initialized for inference")
            return trainer
        except Exception as e:
            logger.error(f"Failed to initialize trainer for inference: {e}")
            return None
    
    def _get_model_checkpoints(self) -> List[str]:
        """Get list of available model checkpoints."""
        checkpoints = list(self.config.model_output_dir.glob("reward_model_epoch_*.safetensors"))
        checkpoint_paths = [str(cp) for cp in checkpoints]
        
        # Also add exported model if it exists
        exported_model = self.config.model_output_dir / "reward_model_for_rlhf.safetensors"
        if exported_model.exists():
            checkpoint_paths.append(str(exported_model))
        
        return checkpoint_paths
    
    def build(self, app_instance: Any) -> None:
        """
        Build the tab's UI components.
        
        Args:
            app_instance: The parent application instance
        """
        with gr.Row():
            with gr.Column():
                # Instructions
                gr.Markdown(
                    """
                    ## Reward Model Inference
                    
                    Test the trained reward model on new images.
                    Upload images to see their predicted reward scores.
                    """
                )
                
                # Model selection
                self.model_dropdown = gr.Dropdown(
                    label="Select Model Checkpoint",
                    choices=self._get_model_checkpoints(),
                    value=None
                )
                self.refresh_models_btn = gr.Button("Refresh Models")
                
                # Load model button
                self.load_model_btn = gr.Button("Load Selected Model")
                self.model_info_text = gr.Markdown("No model loaded")
        
        with gr.Tabs():
            # Single image inference
            with gr.Tab("Single Image"):
                with gr.Row():
                    # Input image
                    self.input_image = gr.Image(
                        label="Input Image",
                        type="pil"
                    )
                    
                    # Optional prompt
                    self.prompt_input = gr.Textbox(
                        label="Optional Prompt",
                        placeholder="Enter a prompt for context...",
                        lines=2
                    )
                    
                    # Prediction output
                    self.prediction_output = gr.Number(
                        label="Predicted Reward",
                        precision=6
                    )
                
                # Predict button
                self.predict_btn = gr.Button("Predict Reward", variant="primary")
            
            # Batch inference
            with gr.Tab("Batch Inference"):
                with gr.Row():
                    # Input folder
                    self.input_folder = gr.Textbox(
                        label="Image Folder Path",
                        placeholder="Path to folder containing images"
                    )
                    
                    # Sample size
                    self.sample_size = gr.Slider(
                        label="Sample Size (max images to process)",
                        minimum=1,
                        maximum=100,
                        value=20,
                        step=1
                    )
                
                # Batch predict button
                self.batch_predict_btn = gr.Button("Analyze Batch", variant="primary")
                
                # Results
                with gr.Row():
                    self.batch_results_plot = gr.Plot(
                        label="Reward Distribution"
                    )
                    
                    self.batch_results_table = gr.Dataframe(
                        label="Top Scoring Images",
                        headers=["Image", "Reward"],
                        datatype=["str", "number"]
                    )
    
    def register_event_handlers(self) -> None:
        """
        Register event handlers for the tab's components.
        """
        self.refresh_models_btn.click(
            self.refresh_models,
            outputs=[self.model_dropdown]
        )
        
        self.load_model_btn.click(
            self.load_model,
            inputs=[self.model_dropdown],
            outputs=[self.model_info_text]
        )
        
        self.predict_btn.click(
            self.predict_reward,
            inputs=[self.input_image, self.prompt_input],
            outputs=[self.prediction_output]
        )
        
        self.batch_predict_btn.click(
            self.batch_predict,
            inputs=[self.input_folder, self.sample_size],
            outputs=[self.batch_results_plot, self.batch_results_table]
        )
    
    def refresh_models(self) -> List[str]:
        """
        Refresh the available model checkpoints.
        
        Returns:
            List of model checkpoint paths
        """
        return self._get_model_checkpoints()
    
    def load_model(self, model_path: str) -> str:
        """
        Load a model checkpoint.
        
        Args:
            model_path: Path to the model checkpoint
            
        Returns:
            Model information text
        """
        if not model_path:
            return "No model selected"
        
        if self.trainer is None:
            return "Trainer not initialized"
        
        try:
            self.trainer.load_model(model_path)
            
            # Try to get metadata
            metadata_path = Path(model_path).with_suffix('').with_suffix('.json')
            if not metadata_path.exists():
                metadata_path = Path(str(model_path).replace('.safetensors', '_metadata.json'))
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                return f"""
                **Model loaded successfully!**
                
                **Path**: {model_path}
                
                **Epoch**: {metadata.get('epoch', 'N/A')}
                
                **Accuracy**: {metadata.get('accuracy', 0):.2%}
                
                **Loss**: {metadata.get('loss', 0):.4f}
                
                **Uses Text**: {metadata.get('use_text', False)}
                """
            else:
                return f"Model loaded from {model_path}"
        except Exception as e:
            return f"Error loading model: {str(e)}"
    
    def predict_reward(self, image: Optional[Image.Image], prompt: Optional[str]) -> Optional[float]:
        """
        Predict the reward for a single image.
        
        Args:
            image: Input image
            prompt: Optional text prompt
            
        Returns:
            Predicted reward score
        """
        if self.trainer is None or not hasattr(self.trainer, 'model'):
            return None
        
        if image is None:
            return None
        
        try:
            # Save image to temporary file
            temp_image_path = Path("temp_inference_image.jpg")
            image.save(temp_image_path)
            
            # Predict reward
            reward = self.trainer.predict_reward(temp_image_path, prompt if prompt else None)
            
            # Clean up
            if temp_image_path.exists():
                temp_image_path.unlink()
            
            return reward
        except Exception as e:
            logger.error(f"Error predicting reward: {e}")
            return None
    
    def batch_predict(self, folder_path: str, max_images: int) -> Tuple[Optional[Figure], Optional[List[List[Any]]]]:
        """
        Run batch inference on a folder of images.
        
        Args:
            folder_path: Path to the folder containing images
            max_images: Maximum number of images to process
            
        Returns:
            Tuple of plot figure and table data
        """
        if self.trainer is None or not hasattr(self.trainer, 'model'):
            return None, None
        
        try:
            path = Path(folder_path)
            if not path.exists():
                return None, None
            
            # Find images
            extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
            images = []
            for ext in extensions:
                images.extend(list(path.glob(f"*{ext}")))
                images.extend(list(path.glob(f"*{ext.upper()}")))
            
            if not images:
                return None, None
            
            # Sample if needed
            if len(images) > max_images:
                images = random.sample(images, max_images)
            
            # Predict rewards
            rewards = self.trainer.batch_predict(images)
            
            # Create plot
            fig = Figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            
            ax.hist(rewards, bins=10, alpha=0.7, color='skyblue')
            ax.set_xlabel('Reward Value')
            ax.set_ylabel('Frequency')
            ax.set_title('Reward Distribution for Batch')
            ax.grid(True, alpha=0.3)
            
            # Add mean and median
            mean_reward = np.mean(rewards)
            median_reward = np.median(rewards)
            
            ax.axvline(mean_reward, color='red', linestyle='dashed', linewidth=1)
            ax.text(mean_reward, 0, f' Mean: {mean_reward:.2f}', 
                    color='red', fontsize=9, va='bottom')
            
            ax.axvline(median_reward, color='green', linestyle='dashed', linewidth=1)
            ax.text(median_reward, 0, f' Median: {median_reward:.2f}', 
                    color='green', fontsize=9, va='top')
            
            fig.tight_layout()
            
            # Create table of top images
            image_rewards = list(zip(images, rewards))
            image_rewards.sort(key=lambda x: x[1], reverse=True)
            
            top_images = []
            for img_path, reward in image_rewards[:10]:  # Top 10
                top_images.append([img_path.name, round(reward, 6)])
            
            return fig, top_images
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            return None, None 