"""
RLHF Optimization tab for the RLHF UI application.

This module contains the implementation of the RLHF Optimization tab, which allows
users to fine-tune generative models using Reinforcement Learning from Human Feedback.
"""

from typing import Any, Dict, Optional, List, Tuple
import os
import logging
import json
from pathlib import Path

import gradio as gr
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.figure import Figure
from PIL import Image

from rlhf_ui.config import AppConfig
from rlhf_ui.models.trainer import RewardModelTrainer
from rlhf_ui.ui.tabs.base import BaseTab

# Get logger for this module
logger = logging.getLogger(__name__)

class RLHFOptimizationTab(BaseTab):
    """
    Tab for RLHF optimization to fine-tune generative models.
    
    This tab allows users to:
    - Configure RLHF optimization parameters
    - Connect reward models to generative models
    - Run fine-tuning and monitor progress
    - Test the optimized model
    """
    
    def __init__(self, config: AppConfig):
        """
        Initialize the RLHF optimization tab.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.trainer = self._init_trainer()
        
        # UI components (will be set during build)
        # Model configuration
        self.reward_model_dropdown = None
        self.target_model_type = None
        self.sd_model_id = None
        self.sd_config = None
        self.custom_model_path = None
        self.custom_model_type = None
        self.custom_config = None
        
        # RLHF parameters
        self.num_iterations = None
        self.learning_rate = None
        self.batch_size = None
        self.kl_weight = None
        
        # Control buttons
        self.start_rlhf_btn = None
        self.stop_rlhf_btn = None
        
        # Status and output
        self.rlhf_status = None
        self.rlhf_progress = None
        self.rlhf_log = None
        self.rlhf_metrics_plot = None
        
        # Test components
        self.test_prompt = None
        self.test_btn = None
        self.output_image = None
        self.reward_score = None
        
        # RLHF optimizer instance
        self.rlhf_optimizer = None
        self.optimization_running = False
    
    def _init_trainer(self) -> Optional[RewardModelTrainer]:
        """Initialize the reward model trainer."""
        try:
            preference_file = self.config.output_folder / "preferences.csv"
            trainer = RewardModelTrainer(
                preference_file=preference_file,
                image_folder=self.config.image_folder,
                model_output_dir=self.config.model_output_dir
            )
            logger.info("Reward model trainer initialized for RLHF")
            return trainer
        except Exception as e:
            logger.error(f"Failed to initialize trainer for RLHF: {e}")
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
                    ## RLHF Optimization
                    
                    Optimize a model using Reinforcement Learning from Human Feedback.
                    Connect your generative model and optimize it using the trained reward model.
                    """
                )
                
                # Model selection
                with gr.Group():
                    gr.Markdown("### Model Configuration")
                    
                    # Reward model selection
                    self.reward_model_dropdown = gr.Dropdown(
                        label="Select Reward Model",
                        choices=self._get_model_checkpoints(),
                        value=None
                    )
                    
                    # Target model type
                    self.target_model_type = gr.Dropdown(
                        label="Target Model Type",
                        choices=["Stable Diffusion", "Custom"],
                        value="Stable Diffusion"
                    )
                    
                    # Target model config
                    self.sd_config = gr.Group(visible=True)
                    with self.sd_config:
                        self.sd_model_id = gr.Textbox(
                            label="Stable Diffusion Model ID",
                            value="runwayml/stable-diffusion-v1-5",
                            placeholder="Enter model ID from Hugging Face (e.g., runwayml/stable-diffusion-v1-5)"
                        )
                    
                    self.custom_config = gr.Group(visible=False)
                    with self.custom_config:
                        self.custom_model_path = gr.Textbox(
                            label="Custom Model Path",
                            placeholder="Path to your custom model"
                        )
                        self.custom_model_type = gr.Textbox(
                            label="Custom Model Type",
                            placeholder="E.g., diffusers, pytorch_model"
                        )
                
                # RLHF parameters
                with gr.Group():
                    gr.Markdown("### RLHF Parameters")
                    
                    self.num_iterations = gr.Slider(
                        label="Number of Iterations",
                        minimum=1,
                        maximum=100,
                        value=10,
                        step=1
                    )
                    
                    self.learning_rate = gr.Slider(
                        label="Learning Rate",
                        minimum=1e-7,
                        maximum=1e-4,
                        value=1e-6,
                        step=1e-7
                    )
                    
                    self.batch_size = gr.Slider(
                        label="Batch Size",
                        minimum=1,
                        maximum=16,
                        value=4,
                        step=1
                    )
                    
                    self.kl_weight = gr.Slider(
                        label="KL Divergence Weight",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.1,
                        step=0.01,
                        info="Controls how much the model can deviate from its initial behavior"
                    )
                
                # Start optimization button
                self.start_rlhf_btn = gr.Button("Start RLHF Optimization", variant="primary")
                self.stop_rlhf_btn = gr.Button("Stop Optimization")
            
            with gr.Column():
                # Optimization status
                self.rlhf_status = gr.Markdown("Not started")
                self.rlhf_progress = gr.Progress()
                
                # Log output
                self.rlhf_log = gr.Textbox(
                    label="Optimization Log",
                    placeholder="Optimization log will appear here...",
                    lines=15,
                    max_lines=30,
                    interactive=False
                )
                
                # Metrics plots
                self.rlhf_metrics_plot = gr.Plot(
                    label="Optimization Metrics"
                )
        
        # Output preview
        with gr.Row():
            with gr.Column():
                # Prompt for testing
                self.test_prompt = gr.Textbox(
                    label="Test Prompt",
                    placeholder="Enter a prompt to test the optimized model...",
                    lines=2
                )
                
                # Generate button
                self.test_btn = gr.Button("Generate Sample")
            
            with gr.Column():
                # Output image
                self.output_image = gr.Image(
                    label="Generated Output",
                    type="pil"
                )
                
                # Reward score
                self.reward_score = gr.Number(
                    label="Reward Score",
                    precision=6
                )
    
    def register_event_handlers(self) -> None:
        """
        Register event handlers for the tab's components.
        """
        self.target_model_type.change(
            self.toggle_model_config,
            inputs=[self.target_model_type],
            outputs=[self.sd_config, self.custom_config]
        )
        
        self.start_rlhf_btn.click(
            self.start_rlhf_optimization,
            inputs=[
                self.reward_model_dropdown,
                self.target_model_type,
                self.sd_model_id,
                self.custom_model_path,
                self.custom_model_type,
                self.num_iterations,
                self.learning_rate,
                self.batch_size,
                self.kl_weight
            ],
            outputs=[
                self.rlhf_log,
                self.rlhf_metrics_plot,
                self.rlhf_status
            ]
        )
        
        self.test_btn.click(
            self.generate_sample,
            inputs=[self.test_prompt],
            outputs=[self.output_image, self.reward_score]
        )
    
    def toggle_model_config(self, model_type: str) -> Tuple[gr.Group, gr.Group]:
        """
        Toggle visibility of model configuration groups.
        
        Args:
            model_type: Type of target model ("Stable Diffusion" or "Custom")
            
        Returns:
            Updated visibility states for the config groups
        """
        if model_type == "Stable Diffusion":
            return gr.Group.update(visible=True), gr.Group.update(visible=False)
        else:
            return gr.Group.update(visible=False), gr.Group.update(visible=True)
    
    def start_rlhf_optimization(
        self, reward_model_path: str, model_type: str, sd_model_id: str, 
        custom_model_path: str, custom_model_type: str, num_iterations: int,
        learning_rate: float, batch_size: int, kl_weight: float, 
        progress=gr.Progress()
    ) -> Tuple[str, Optional[Figure], str]:
        """
        Start RLHF optimization process.
        
        Args:
            reward_model_path: Path to the reward model
            model_type: Type of target model
            sd_model_id: Stable Diffusion model ID
            custom_model_path: Path to custom model
            custom_model_type: Type of custom model
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate
            batch_size: Batch size
            kl_weight: KL divergence weight
            progress: Gradio progress instance
            
        Returns:
            Tuple of log text, metrics plot, and status message
        """
        if not reward_model_path:
            return "Error: No reward model selected", None, "Error: No reward model selected"
        
        # Initialize log
        log_entries = []
        log_entries.append("Initializing RLHF optimization...")
        
        try:
            # Initialize reward model
            if self.trainer is None:
                self.trainer = self._init_trainer()
            
            if self.trainer is None:
                return "Error: Failed to initialize trainer", None, "Error: Failed to initialize trainer"
            
            # Load reward model
            log_entries.append(f"Loading reward model from {reward_model_path}")
            self.trainer.load_model(reward_model_path)
            
            # Initialize target model
            if model_type == "Stable Diffusion":
                log_entries.append(f"Loading Stable Diffusion model: {sd_model_id}")
                
                # Import here to avoid loading dependencies unless needed
                try:
                    from diffusers import StableDiffusionPipeline # type: ignore
                    import torch
                    
                    target_model = StableDiffusionPipeline.from_pretrained(
                        sd_model_id,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    )
                    
                    if torch.cuda.is_available():
                        target_model = target_model.to("cuda")
                        
                    log_entries.append("Model loaded successfully")
                except Exception as e:
                    error_msg = f"Error loading Stable Diffusion model: {str(e)}"
                    log_entries.append(error_msg)
                    return "\n".join(log_entries), None, error_msg
            else:
                log_entries.append(f"Loading custom model from {custom_model_path}")
                # Custom model loading logic would go here
                return "Custom model support not implemented yet", None, "Custom model support not implemented yet"
            
            # Initialize RLHF optimizer
            from rlhf_ui.rlhf.optimizer import RLHFOptimizer # type: ignore
            
            log_entries.append("Initializing RLHF optimizer...")
            self.rlhf_optimizer = RLHFOptimizer(
                target_model=target_model,
                reward_model=self.trainer.model,
                learning_rate=learning_rate,
                kl_weight=kl_weight
            )
            
            # Setup metrics tracking
            reward_values = []
            kl_div_values = []
            loss_values = []
            
            # Training loop
            log_entries.append(f"Starting optimization with {num_iterations} iterations")
            
            for iteration in range(num_iterations):
                progress((iteration + 1) / num_iterations)
                log_entries.append(f"Iteration {iteration + 1}/{num_iterations}")
                
                # Run a single optimization step
                metrics = self.rlhf_optimizer.optimization_step(
                    batch_size=batch_size
                )
                
                # Log metrics
                reward_values.append(metrics['reward'])
                kl_div_values.append(metrics['kl_div'])
                loss_values.append(metrics['loss'])
                
                log_entries.append(
                    f"  Reward: {metrics['reward']:.4f}, "
                    f"KL Div: {metrics['kl_div']:.4f}, "
                    f"Loss: {metrics['loss']:.4f}"
                )
                
                # Create metrics plot
                if (iteration + 1) % 5 == 0 or iteration == num_iterations - 1:
                    fig = Figure(figsize=(10, 6))
                    
                    # Plot reward
                    ax1 = fig.add_subplot(311)
                    ax1.plot(range(1, len(reward_values) + 1), reward_values, 'b-')
                    ax1.set_ylabel('Reward')
                    ax1.set_title('RLHF Optimization Metrics')
                    ax1.grid(True, alpha=0.3)
                    
                    # Plot KL divergence
                    ax2 = fig.add_subplot(312)
                    ax2.plot(range(1, len(kl_div_values) + 1), kl_div_values, 'r-')
                    ax2.set_ylabel('KL Divergence')
                    ax2.grid(True, alpha=0.3)
                    
                    # Plot combined loss
                    ax3 = fig.add_subplot(313)
                    ax3.plot(range(1, len(loss_values) + 1), loss_values, 'g-')
                    ax3.set_xlabel('Iteration')
                    ax3.set_ylabel('Loss')
                    ax3.grid(True, alpha=0.3)
                    
                    fig.tight_layout()
            
            log_entries.append("Optimization completed successfully!")
            
            # Save optimized model
            save_path = self.config.model_output_dir / "rlhf_optimized_model"
            save_path.mkdir(exist_ok=True, parents=True)
            
            log_entries.append(f"Saving optimized model to {save_path}")
            target_model.save_pretrained(save_path)
            
            # Save optimization metrics
            metrics_path = save_path / "optimization_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump({
                    'reward': reward_values,
                    'kl_div': kl_div_values,
                    'loss': loss_values,
                    'params': {
                        'num_iterations': num_iterations,
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'kl_weight': kl_weight
                    }
                }, f, indent=2)
            
            return "\n".join(log_entries), fig, "Optimization completed! Model saved."
        
        except Exception as e:
            error_msg = f"Error during RLHF optimization: {str(e)}"
            log_entries.append(error_msg)
            return "\n".join(log_entries), None, error_msg
    
    def generate_sample(self, prompt: str) -> Tuple[Optional[Image.Image], Optional[float]]:
        """
        Generate a sample using the optimized model.
        
        Args:
            prompt: Text prompt for image generation
            
        Returns:
            Tuple of generated image and reward score
        """
        if self.rlhf_optimizer is None or not hasattr(self.rlhf_optimizer, 'target_model'):
            return None, None
        
        try:
            # Generate image
            image = self.rlhf_optimizer.generate(prompt)
            
            # Get reward score
            score = self.rlhf_optimizer.compute_reward(image, prompt)
            
            return image, score
        except Exception as e:
            logger.error(f"Error generating sample: {e}")
            return None, None 