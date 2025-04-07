"""
Reward Model tab for the RLHF UI application.

This module contains the implementation of the Reward Model tab, which allows
users to train reward models based on collected preferences.
"""

from typing import Any, Dict, Optional, List, Tuple
import os
import logging
import webbrowser
import threading
import time
from pathlib import Path

import gradio as gr
import torch
import wandb
import plotly.graph_objects as go

from rlhf_ui.config import AppConfig
from rlhf_ui.models.trainer import RewardModelTrainer
from rlhf_ui.visualization import (
    init_wandb, 
    finish_run, 
    get_wandb_url,
    create_embedded_iframe,
    generate_live_metrics_chart,
    generate_separate_metric_charts
)
from rlhf_ui.ui.tabs.base import BaseTab

# Get logger for this module
logger = logging.getLogger(__name__)

class RewardModelTab(BaseTab):
    """
    Tab for training reward models from collected preference data.
    
    This tab allows users to:
    - Configure and train reward models
    - Monitor training progress
    - Export models for inference and RLHF
    """
    
    def __init__(self, config: AppConfig):
        """
        Initialize the reward model training tab.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.trainer = self._init_trainer()
        
        # UI components (will be set during build)
        self.train_dataset_path = None
        self.epochs_slider = None
        self.batch_size_slider = None
        self.learning_rate_slider = None
        self.model_type_dropdown = None
        self.wandb_project_name = None
        self.wandb_run_name = None
        self.train_btn = None
        self.stop_btn = None
        self.export_btn = None
        self.wandb_link = None
        self.model_output_dir = None
        self.training_log = None
        self.training_progress = None
        self.training_status = None
        self.data_info_text = None
        self.refresh_data_btn = None
        self.model_info_text = None
        self.dashboard_iframe = None
        self.open_dashboard_btn = None
        self.use_text_checkbox = None
        self.use_gpu_checkbox = None
        self.metrics_plot = None
        self.refresh_metrics_btn = None
        self.accuracy_plot = None  # New plot for accuracy metrics
        self.loss_plot = None      # New plot for loss metrics
        
        # State variables
        self.is_training = False
        self.current_wandb_run = None
        self.training_thread = None
        self.stop_training = False
        self.wandb_run_id = None
        
        # Monitoring thread
        self.monitor_thread = None
        self.stop_monitoring = False
    
    def _init_trainer(self) -> Optional[RewardModelTrainer]:
        """Initialize the reward model trainer."""
        try:
            preference_file = self.config.output_folder / "preferences.csv"
            trainer = RewardModelTrainer(
                preference_file=preference_file,
                image_folder=self.config.image_folder,
                model_output_dir=self.config.model_output_dir
            )
            logger.info("Reward model trainer initialized")
            return trainer
        except Exception as e:
            logger.error(f"Failed to initialize trainer: {e}")
            return None
    
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
                    ## Train Reward Model
                    
                    Train a reward model based on the collected preferences.
                    You need at least 10 preference pairs to start training.
                    """
                )
                
                # Training parameters
                with gr.Group():
                    gr.Markdown("### Training Parameters")
                    
                    self.epochs_slider = gr.Number(
                        label="Training Epochs",
                        value=20,
                        minimum=1,
                        maximum=100,
                        step=1,
                    )
                    
                    self.learning_rate_slider = gr.Number(
                        label="Learning Rate",
                        value=0.0001,
                        minimum=0.000001,
                        maximum=0.1,
                    )
                    
                    self.batch_size_slider = gr.Number(
                        label="Batch Size",
                        value=8,
                        minimum=1,
                        maximum=64,
                        step=1,
                    )
                    
                    self.use_text_checkbox = gr.Checkbox(
                        label="Use Text/Prompt Data",
                        value=True,
                        info="Include text prompts as part of the model input"
                    )
                    
                    self.use_gpu_checkbox = gr.Checkbox(
                        label="Use GPU for Training",
                        value=True,
                        info="Use CUDA if available"
                    )
                    
                    # Add wandb project and run name inputs
                    self.wandb_project_name = gr.Textbox(
                        label="W&B Project Name",
                        value="rlhf-reward-model",
                        info="Name of the W&B project for tracking"
                    )
                    
                    self.wandb_run_name = gr.Textbox(
                        label="W&B Run Name (optional)",
                        value="",
                        placeholder="Auto-generated if empty",
                        info="Name for this training run"
                    )
                
                # Data info
                with gr.Group():
                    gr.Markdown("### Training Data")
                    
                    self.data_info_text = gr.Markdown("Loading data info...")
                    
                    self.refresh_data_btn = gr.Button("Refresh Data Info")
                
                # Start training
                with gr.Row():
                    self.train_btn = gr.Button("Start Training", variant="primary")
                    self.stop_btn = gr.Button("Stop Training", variant="stop")
                
                # Training log
                self.training_log = gr.Textbox(
                    label="Training Log",
                    lines=10,
                    max_lines=20,
                    interactive=False
                )
                
                # Training status
                self.training_status = gr.Markdown("")
                
                # Export model
                self.export_btn = gr.Button("Export Model for RLHF")
            
            with gr.Column():
                # Dashboard link
                gr.Markdown("### Weights & Biases Dashboard")
                self.wandb_link = gr.Textbox(
                    label="Dashboard URL",
                    interactive=False,
                    placeholder="Training dashboard URL will appear here"
                )
                
                # Open dashboard button
                with gr.Row():
                    self.open_dashboard_btn = gr.Button("Open Dashboard in Browser", size="sm")
                    self.refresh_metrics_btn = gr.Button("Refresh Metrics", size="sm")
                
                # Live metrics plots - display all vertically stacked
                gr.Markdown("### Training Metrics")
                
                # Accuracy plot
                with gr.Group(elem_classes="metric-box"):
                    gr.Markdown("#### Accuracy Metrics", elem_classes="metric-header")
                    self.accuracy_plot = gr.Plot(
                        label="Accuracy Metrics",
                        show_label=False,  # Hide label since we have a markdown header
                        elem_classes="plot-container"
                    )
                
                # Loss plot
                with gr.Group(elem_classes="metric-box"):
                    gr.Markdown("#### Loss Metrics", elem_classes="metric-header")
                    self.loss_plot = gr.Plot(
                        label="Loss Metrics", 
                        show_label=False,  # Hide label since we have a markdown header
                        elem_classes="plot-container"
                    )
                
                # Combined plot for backward compatibility
                with gr.Group(elem_classes="metric-box"):
                    gr.Markdown("#### All Metrics (Combined)", elem_classes="metric-header")
                    self.metrics_plot = gr.Plot(
                        label="All Training Metrics",
                        show_label=False,  # Hide label since we have a markdown header
                        elem_classes="plot-container"
                    )
                
                # Model info
                self.model_info_text = gr.Markdown("No model trained yet")
                
                # Embedded W&B iframe
                self.dashboard_iframe = gr.HTML(
                    """
                    <div style="height: 600px; display: flex; justify-content: center; align-items: center; border: 1px solid #ddd; border-radius: 4px;">
                        <p style="color: #666;">Training visualization will appear here</p>
                    </div>
                    """
                )
    
    def register_event_handlers(self) -> None:
        """
        Register event handlers for the tab's components.
        """
        self.refresh_data_btn.click(
            self.refresh_data_info, 
            outputs=[self.data_info_text]
        )
        
        # Call refresh on load
        self.data_info_text.value = self.refresh_data_info()
        
        self.train_btn.click(
            self._start_training_thread,
            inputs=[
                self.epochs_slider,
                self.learning_rate_slider,
                self.batch_size_slider,
                self.use_text_checkbox,
                self.use_gpu_checkbox,
                self.wandb_project_name,
                self.wandb_run_name
            ],
            outputs=[
                self.training_log,
                self.training_status
            ]
        )
        
        self.stop_btn.click(
            self._stop_training,
            outputs=[
                self.training_status
            ]
        )
        
        self.open_dashboard_btn.click(
            self.open_dashboard,
            inputs=[self.wandb_link],
            outputs=[self.training_status]
        )
        
        # Define a custom refresh metrics function that returns the updated values for all components
        def refresh_metrics_handler():
            combined_fig, separate_figs, html = self._refresh_metrics()
            
            # If there are no separate figures yet, return the combined one for both
            accuracy_fig = separate_figs[0] if separate_figs and len(separate_figs) > 0 else combined_fig
            loss_fig = separate_figs[1] if separate_figs and len(separate_figs) > 1 else combined_fig
            
            return combined_fig, accuracy_fig, loss_fig, html
            
        self.refresh_metrics_btn.click(
            refresh_metrics_handler,
            outputs=[
                self.metrics_plot,
                self.accuracy_plot,
                self.loss_plot,
                self.dashboard_iframe
            ]
        )
        
        self.export_btn.click(
            self.export_model,
            outputs=[self.training_status]
        )
    
    def _start_training_thread(
        self, epochs: int, learning_rate: float, batch_size: int, use_text: bool, use_gpu: bool,
        wandb_project: str, wandb_run_name: str
    ) -> Tuple[str, str]:
        """
        Start training in a separate thread to keep the UI responsive.
        
        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            use_text: Whether to use text embeddings
            use_gpu: Whether to use GPU
            wandb_project: W&B project name
            wandb_run_name: W&B run name
            
        Returns:
            Tuple of training log and status
        """
        # Initialize logging and status
        self.progress_log = ["Initializing training..."]
        
        # Initialize progress tracking
        log_text = "Starting training in a separate thread..."
        status_text = "Training in progress..."
        
        # Reset plots
        empty_plot = go.Figure()
        empty_plot.update_layout(
            title="No metrics available yet",
            title_x=0.5,
            height=320,
            font=dict(family="Arial, sans-serif", size=12),
            xaxis_title="Training Step",
            yaxis_title="Value",
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.2)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.2)'),
            annotations=[
                dict(
                    text="Training is starting...<br>Metrics will appear here",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    font=dict(size=14)
                )
            ]
        )
        
        self.metrics_plot.value = empty_plot
        self.accuracy_plot.value = empty_plot
        self.loss_plot.value = empty_plot
        
        # Reset dashboard
        self.dashboard_iframe.value = """
        <div style="height: 600px; display: flex; justify-content: center; align-items: center; border: 1px solid #ddd; border-radius: 4px;">
            <p style="color: #666;">Training is starting... Dashboard will appear shortly</p>
        </div>
        """
        
        # Initialize W&B if not already running
        if wandb.run is not None:
            finish_run()
        
        # Initialize a new run with the specified project and run name
        init_wandb(
            project_name=wandb_project,
            experiment_name=wandb_run_name
        )
        
        # Set state flag
        self.is_training = True
        self.stop_training = False
        
        # Create and start training thread
        self.training_thread = threading.Thread(
            target=self._run_training,
            args=(epochs, learning_rate, batch_size, use_text, use_gpu, wandb_project, wandb_run_name)
        )
        self.training_thread.start()
        
        # Start monitoring thread
        self._start_monitoring()
        
        return log_text, status_text
    
    def _run_training(
        self, 
        epochs: int,
        learning_rate: float,
        batch_size: int,
        use_text: bool,
        use_gpu: bool,
        wandb_project: str,
        wandb_run_name: Optional[str] = None
    ) -> None:
        """
        Run the training process in a separate thread.
        
        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            use_text: Whether to use text inputs
            use_gpu: Whether to use GPU
            wandb_project: W&B project name
            wandb_run_name: Optional W&B run name
        """
        # Set training status
        self.is_training = True
        
        try:
            # Setup device
            if use_gpu and torch.cuda.is_available():
                torch.cuda.set_device(0)
            else:
                # Force CPU
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
            
            # Setup progress tracking
            self.progress_log = []
            
            # Progress callback
            def update_progress(epoch, progress_pct, loss, accuracy):
                # Add to log
                msg = f"Epoch {epoch+1}/{epochs}: {progress_pct}% complete - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%"
                self.progress_log.append(msg)
                
                # Update UI elements (cannot update directly from thread)
                # They will be updated by the monitoring thread
            
                # Check if training should be stopped
                if self.stop_training:
                    raise InterruptedError("Training stopped by user")
            
            # Start training
            text_embedding_size = 768 if use_text else 0
            best_model_path = self.trainer.train(
                epochs=epochs,
                lr=learning_rate,
                batch_size=batch_size,
                text_embedding_size=text_embedding_size,
                progress_callback=update_progress,
                use_wandb=True,
                wandb_project=wandb_project,
                wandb_run_name=wandb_run_name
            )
            
            # Save wandb run ID if available
            if wandb.run is not None:
                self.wandb_run_id = wandb.run.id
                self.current_wandb_run = wandb.run
            
            # Add completion message
            self.progress_log.append("Training completed successfully!")
            self.progress_log.append(f"Best model saved to: {best_model_path}")
            
        except InterruptedError as e:
            # User stopped training
            self.progress_log.append(f"Training stopped by user")
            
            # Finish wandb run
            if wandb.run is not None:
                finish_run()
                
        except Exception as e:
            # Handle other errors
            error_msg = f"Error during training: {str(e)}"
            self.progress_log.append(error_msg)
            logger.error(error_msg)
            
            # Finish wandb run
            if wandb.run is not None:
                finish_run()
        
        finally:
            # Reset training status
            self.is_training = False
    
    def _stop_training(self) -> str:
        """
        Stop the training process.
        
        Returns:
            Status message
        """
        if not self.is_training:
            return "No training in progress"
        
        # Set stop flag
        self.stop_training = True
        
        return "Stopping training... Please wait for current epoch to complete."
    
    def _start_monitoring(self) -> None:
        """Start a thread to monitor training progress and update UI."""
        if self.monitor_thread is not None and self.monitor_thread.is_alive():
            # Already monitoring
            return
        
        self.stop_monitoring = False
        
        # Create and start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_training,
            daemon=True
        )
        self.monitor_thread.start()
    
    def _monitor_training(self) -> None:
        """Monitor training and update UI components."""
        last_log_size = 0
        
        while not self.stop_monitoring:
            try:
                # Store updates to be applied in the main thread later
                updates = {}
                
                # Update log if there are new entries
                if hasattr(self, 'progress_log') and len(self.progress_log) > last_log_size:
                    # Get new log entries
                    new_entries = self.progress_log[last_log_size:]
                    last_log_size = len(self.progress_log)
                    
                    # Store update for log component
                    log_text = "\n".join(self.progress_log)
                    if hasattr(self, 'training_log') and self.training_log is not None:
                        updates['training_log'] = log_text
                
                # Update wandb URL if available
                if wandb.run is not None:
                    # Get URL
                    url = get_wandb_url()
                    if url:
                        # Store update for URL
                        updates['wandb_url'] = url
                        
                        # Create iframe HTML
                        iframe_html = create_embedded_iframe(url)
                        updates['dashboard_iframe'] = iframe_html
                
                # Update metrics plot if we have a run ID
                if self.wandb_run_id is not None:
                    try:
                        combined_fig, separate_figs, dashboard_html = self._refresh_metrics()
                        updates['metrics_plot'] = combined_fig
                        
                        # Update separate plots if available
                        if separate_figs:
                            if len(separate_figs) > 0:
                                updates['accuracy_plot'] = separate_figs[0]
                            if len(separate_figs) > 1:
                                updates['loss_plot'] = separate_figs[1]
                        
                        if 'dashboard_iframe' not in updates:  # Only update if not already set
                            updates['dashboard_iframe'] = dashboard_html
                    except Exception as e:
                        logger.error(f"Error refreshing metrics: {e}")
                
                # Check if training is done
                if not self.is_training:
                    updates['training_status'] = "Training completed"
                    
                    # If training just finished, do a final metrics refresh
                    if 'metrics_plot' not in updates:
                        try:
                            combined_fig, separate_figs, dashboard_html = self._refresh_metrics()
                            updates['metrics_plot'] = combined_fig
                            
                            # Update separate plots if available
                            if separate_figs:
                                if len(separate_figs) > 0:
                                    updates['accuracy_plot'] = separate_figs[0]
                                if len(separate_figs) > 1:
                                    updates['loss_plot'] = separate_figs[1]
                            
                            if 'dashboard_iframe' not in updates:
                                updates['dashboard_iframe'] = dashboard_html
                        except Exception as e:
                            logger.error(f"Error in final metrics refresh: {e}")
                    
                    # Stop monitoring after applying updates
                    self.stop_monitoring = True
                
                # Apply all updates to UI components in the main thread
                # Gradio 5.x uses values directly instead of update() method
                if 'training_log' in updates and hasattr(self, 'training_log'):
                    self.training_log.value = updates['training_log']
                
                if 'wandb_url' in updates and hasattr(self, 'wandb_link'):
                    self.wandb_link.value = updates['wandb_url']
                
                if 'dashboard_iframe' in updates and hasattr(self, 'dashboard_iframe'):
                    self.dashboard_iframe.value = updates['dashboard_iframe']
                
                if 'metrics_plot' in updates and hasattr(self, 'metrics_plot'):
                    self.metrics_plot.value = updates['metrics_plot']
                
                if 'accuracy_plot' in updates and hasattr(self, 'accuracy_plot'):
                    self.accuracy_plot.value = updates['accuracy_plot']
                
                if 'loss_plot' in updates and hasattr(self, 'loss_plot'):
                    self.loss_plot.value = updates['loss_plot']
                
                if 'training_status' in updates and hasattr(self, 'training_status'):
                    self.training_status.value = updates['training_status']
                
                # Sleep to avoid high CPU usage
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error in monitoring thread: {e}")
                time.sleep(5)  # Longer sleep on error
        
        logger.info("Monitoring thread stopped")
        # Clear flags
        self.stop_monitoring = False
    
    def _refresh_metrics(self) -> Tuple[Optional[go.Figure], List[Optional[go.Figure]], str]:
        """
        Refresh the metrics plots and dashboard iframe.
        
        Returns:
            Tuple containing:
            - The combined metrics figure
            - List of separate metric figures
            - HTML for the dashboard iframe
        """
        try:
            # Get the current wandb URL
            wandb_url = get_wandb_url()
            if not wandb_url:
                self.training_log.value += "\nNo active W&B run found."
                return None, [], ""
            
            run_id = wandb_url.split("/")[-1]
            metric_keys = [
                "train/loss",
                "train/accuracy",
                "train/running_loss",
                "train/running_accuracy",
                "train/best_accuracy"
            ]
                
            # Generate the combined metrics plot
            combined_fig = generate_live_metrics_chart(
                run_id=run_id,
                metric_keys=metric_keys
            )
            
            # Customize combined figure for better display
            if combined_fig is not None:
                combined_fig.update_layout(
                    height=380,  # Slightly taller than individual plots
                    margin=dict(l=40, r=40, t=50, b=40),
                    title="All Training Metrics",
                    title_x=0.5,  # Center the title
                    title_font=dict(size=16),
                    font=dict(family="Arial, sans-serif", size=12),
                    plot_bgcolor='rgba(250,250,250,0.5)',  # Very light background
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="center",
                        x=0.5,
                        font=dict(size=11)
                    )
                )
            
            # Generate separate plots for different metric groups
            separate_figs = generate_separate_metric_charts(
                run_id=run_id,
                metric_keys=metric_keys
            )
            
            # Customize each figure for better display in the vertical layout
            if separate_figs:
                for i, fig in enumerate(separate_figs):
                    if fig is not None:
                        # Set a consistent size and styling
                        fig.update_layout(
                            height=350,  # Slightly shorter than combined plot
                            margin=dict(l=40, r=40, t=40, b=40),
                            title_x=0.5,  # Center the title
                            legend=dict(
                                orientation="h",  # Horizontal legend
                                yanchor="bottom",
                                y=1.02,  # Positioned above the plot
                                xanchor="center",
                                x=0.5  # Centered
                            ),
                            plot_bgcolor='rgba(240,240,240,0.3)'  # Light grey background
                        )
            
            if combined_fig is None and not separate_figs:
                self.training_log.value += "\nNo metrics available yet. Please wait for training to progress."
                return None, [], ""
            
            # Generate the dashboard iframe
            iframe_html = create_embedded_iframe(wandb_url)
            
            return combined_fig, separate_figs, iframe_html
            
        except Exception as e:
            error_msg = f"Error refreshing metrics: {str(e)}"
            self.training_log.value += f"\n{error_msg}"
            logger.error(error_msg)
            return None, [], ""
    
    def refresh_data_info(self) -> str:
        """
        Refresh the data information display.
        
        Returns:
            Data information text
        """
        try:
            # Reset trainer to reload data
            self.trainer = self._init_trainer()
            
            if self.trainer is None:
                return "Failed to initialize trainer"
            
            # Get data summary
            preference_count = len(self.trainer.preferences_df)
            
            # Count checkpoints
            checkpoints = list(self.config.model_output_dir.glob("reward_model_epoch_*.safetensors"))
            
            return f"""
            **Preference Records**: {preference_count}
            
            **Checkpoints**: {len(checkpoints)} found
            
            **Status**: {"Ready to train" if preference_count > 10 else "Need more preferences (at least 10 recommended)"}
            """
        except Exception as e:
            return f"Error refreshing data info: {str(e)}"
    
    def start_training(
        self, epochs: int, learning_rate: float, batch_size: int, 
        use_text: bool, use_gpu: bool, progress=gr.Progress()
    ) -> Tuple[str, str, str, str]:
        """
        Start training the reward model.
        
        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            use_text: Whether to include text/prompt data
            use_gpu: Whether to use GPU for training
            progress: Gradio progress instance
            
        Returns:
            Tuple of training log, wandb URL, model info, and dashboard HTML
        """
        if self.trainer is None:
            return "Trainer not initialized", "", "No model available", ""
        
        # Setup device
        if use_gpu and torch.cuda.is_available():
            torch.cuda.set_device(0)
        else:
            # Force CPU
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        # Initialize log
        log_entries = []
        
        try:
            # Check if we have enough data
            if len(self.trainer.preferences_df) < 2:
                return "Not enough preference data for training. Need at least 2 records.", "", "No model available", ""
            
            log_entries.append(f"Starting training with {len(self.trainer.preferences_df)} preference records")
            log_entries.append(f"Parameters: epochs={epochs}, lr={learning_rate}, batch_size={batch_size}, use_text={use_text}")
            log_entries.append(f"Using {'GPU' if use_gpu and torch.cuda.is_available() else 'CPU'} for training")
            
            # Setup progress callback
            def update_progress(epoch, progress_percent, loss, accuracy):
                if progress_percent == 0:
                    log_entries.append(f"Epoch {epoch+1}/{epochs} started")
                
                # Only update at 0, 25, 50, 75, 100% to avoid too many updates
                if progress_percent in [0, 25, 50, 75, 100]:
                    progress((epoch * 100 + progress_percent) / (epochs * 100))
                
                if progress_percent == 100:
                    log_entries.append(f"Epoch {epoch+1} completed: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
            
            # Start training with W&B enabled
            text_embedding_size = 768 if use_text else 0
            best_model_path = self.trainer.train(
                epochs=epochs,
                lr=learning_rate,
                batch_size=batch_size,
                text_embedding_size=text_embedding_size,
                progress_callback=update_progress,
                use_wandb=True
            )
            
            log_entries.append("Training completed successfully!")
            log_entries.append(f"Best model saved to: {best_model_path}")
            
            # Save wandb run ID if available
            if wandb.run is not None:
                self.wandb_run_id = wandb.run.id
                self.current_wandb_run = wandb.run
            
            # Get W&B dashboard URL
            wandb_url = ""
            if wandb.run is not None:
                wandb_url = wandb.run.get_url()
                log_entries.append(f"W&B Dashboard: {wandb_url}")
            
            # Create dashboard iframe HTML
            dashboard_html = create_embedded_iframe(wandb_url)
            
            # Get model info
            model_info = f"""
            **Training completed!**
            
            **Final Loss**: {self.trainer.epoch_losses[-1]:.4f}
            
            **Final Accuracy**: {self.trainer.epoch_accs[-1]:.2f}% 
            
            **Best Accuracy**: {self.trainer.best_accuracy:.2f}% (Epoch {self.trainer.best_epoch + 1})
            
            **Best Model**: {Path(best_model_path).name if best_model_path else "Not available"}
            """
            
            return "\n".join(log_entries), wandb_url, model_info, dashboard_html

        except Exception as e:
            error_msg = f"Error during training: {str(e)}"
            log_entries.append(error_msg)
            return "\n".join(log_entries), "", error_msg, ""
    
    def open_dashboard(self, url: str) -> str:
        """
        Open the W&B dashboard in a browser tab.
        
        Args:
            url: W&B dashboard URL
            
        Returns:
            Status message
        """
        if url:
            try:
                webbrowser.open(url)
                return "Dashboard opened in browser"
            except Exception as e:
                return f"Error opening dashboard: {e}"
        else:
            return "No dashboard URL available"
    
    def export_model(self) -> str:
        """
        Export the trained model for RLHF.
        
        Returns:
            Status message
        """
        if self.trainer is None:
            return "Trainer not initialized"
        
        try:
            # Check for model checkpoints
            checkpoints = list(self.config.model_output_dir.glob("reward_model_epoch_*.safetensors"))
            if not checkpoints:
                return "No trained models found. Please train the model first."
            
            # Export model
            export_path = self.trainer.export_for_rlhf()
            
            return f"""
            **Model exported successfully!**
            
            **Export path**: {export_path}
            
            The model is now ready for use in RLHF fine-tuning.
            """
        except Exception as e:
            return f"Error exporting model: {str(e)}" 