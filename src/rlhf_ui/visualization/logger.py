"""
Visualization and experiment tracking with Weights & Biases.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import torch
import numpy as np
import wandb

logger = logging.getLogger(__name__)


def init_wandb(
    project_name: str = "rlhf-ui",
    experiment_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None
) -> None:
    """
    Initialize a new Weights & Biases run for experiment tracking.
    
    Args:
        project_name: Name of the W&B project 
        experiment_name: Optional name for the experiment run
        config: Configuration parameters to track
        tags: Optional list of tags for the run
    """
    try:
        # Default config if none provided
        if config is None:
            config = {}
            
        # Initialize a new W&B run
        wandb.init(
            project=project_name,
            name=experiment_name,
            config=config,
            tags=tags,
            reinit=True
        )
        
        logger.info(f"Initialized Weights & Biases run: {wandb.run.name}")
    except Exception as e:
        logger.error(f"Failed to initialize Weights & Biases: {e}")
        logger.warning("Experiment tracking will not be available")


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """
    Log metrics to Weights & Biases.
    
    Args:
        metrics: Dictionary of metric names and values
        step: Optional step number for the metrics
    """
    if not _is_wandb_active():
        return
        
    try:
        wandb.log(metrics, step=step)
    except Exception as e:
        logger.error(f"Failed to log metrics to W&B: {e}")


def log_image_comparison(
    image1: Union[str, Path, np.ndarray],
    image2: Union[str, Path, np.ndarray],
    preference: int,
    prompt: Optional[str] = None,
    step: Optional[int] = None
) -> None:
    """
    Log a pair of compared images with preference information.
    
    Args:
        image1: First image (file path or numpy array)
        image2: Second image (file path or numpy array)
        preference: Preference value (1 for image1, 2 for image2, 0 for tie)
        prompt: Optional prompt used for the comparison
        step: Optional step number
    """
    if not _is_wandb_active():
        return
        
    try:
        # Prepare caption based on preference
        if preference == 1:
            caption = "Preferred ← | →"
        elif preference == 2:
            caption = "| ← Preferred →"
        else:
            caption = "Equal preference"
            
        # Add prompt to caption if available
        if prompt:
            caption = f"{caption}\nPrompt: {prompt}"
            
        # Log the images as a comparison
        wandb.log({
            "Image Comparison": wandb.Image(
                _prepare_comparison_image(image1, image2),
                caption=caption
            )
        }, step=step)
    except Exception as e:
        logger.error(f"Failed to log image comparison to W&B: {e}")


def log_reward_predictions(
    images: List[Union[str, Path, np.ndarray]],
    rewards: List[float],
    step: Optional[int] = None
) -> None:
    """
    Log a batch of images with their predicted rewards.
    
    Args:
        images: List of images
        rewards: List of corresponding reward predictions
        step: Optional step number
    """
    if not _is_wandb_active():
        return
        
    try:
        # Create a table for the results
        columns = ["image", "reward"]
        data = []
        
        for img, reward in zip(images, rewards):
            # Convert image to wandb.Image
            if isinstance(img, (str, Path)):
                wandb_img = wandb.Image(str(img))
            else:
                wandb_img = wandb.Image(img)
                
            data.append([wandb_img, reward])
            
        # Log the table
        wandb.log({
            "Reward Predictions": wandb.Table(data=data, columns=columns)
        }, step=step)
    except Exception as e:
        logger.error(f"Failed to log reward predictions to W&B: {e}")


def log_model(
    model: torch.nn.Module,
    model_name: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log a trained model to Weights & Biases.
    
    Args:
        model: PyTorch model to save
        model_name: Name for the saved model
        metadata: Optional metadata to associate with the model
    """
    if not _is_wandb_active():
        return
        
    try:
        # Create artifact for the model
        artifact = wandb.Artifact(
            name=model_name,
            type="model",
            metadata=metadata
        )
        
        # Save model to a temporary file
        model_path = os.path.join(wandb.run.dir, f"{model_name}.pt")
        torch.save(model.state_dict(), model_path)
        
        # Add model file to artifact
        artifact.add_file(model_path)
        
        # Log artifact
        wandb.log_artifact(artifact)
        logger.info(f"Logged model {model_name} to W&B")
    except Exception as e:
        logger.error(f"Failed to log model to W&B: {e}")


def finish_run() -> None:
    """
    Finish the current W&B run.
    """
    if not _is_wandb_active():
        return
        
    try:
        wandb.finish()
        logger.info("Finished W&B run")
    except Exception as e:
        logger.error(f"Error finishing W&B run: {e}")


def _is_wandb_active() -> bool:
    """Check if W&B is currently active."""
    return wandb.run is not None


def _prepare_comparison_image(
    image1: Union[str, Path, np.ndarray],
    image2: Union[str, Path, np.ndarray]
) -> np.ndarray:
    """
    Prepare a side-by-side comparison image.
    
    Args:
        image1: First image
        image2: Second image
        
    Returns:
        np.ndarray: Combined image
    """
    # Load images if they are file paths
    import PIL.Image
    
    if isinstance(image1, (str, Path)):
        img1 = np.array(PIL.Image.open(str(image1)))
    else:
        img1 = image1
        
    if isinstance(image2, (str, Path)):
        img2 = np.array(PIL.Image.open(str(image2)))
    else:
        img2 = image2
        
    # Ensure both images have the same height
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    target_height = max(h1, h2)
    
    if h1 != target_height:
        # Resize image1 to match target height while maintaining aspect ratio
        new_w1 = int(w1 * (target_height / h1))
        img1 = PIL.Image.fromarray(img1).resize((new_w1, target_height))
        img1 = np.array(img1)
        
    if h2 != target_height:
        # Resize image2 to match target height while maintaining aspect ratio
        new_w2 = int(w2 * (target_height / h2))
        img2 = PIL.Image.fromarray(img2).resize((new_w2, target_height))
        img2 = np.array(img2)
    
    # Add a vertical separator
    separator_width = 5
    separator = np.ones((target_height, separator_width, 3), dtype=np.uint8) * 255
    
    # Concatenate horizontally
    combined_image = np.concatenate([img1, separator, img2], axis=1)
    
    return combined_image 