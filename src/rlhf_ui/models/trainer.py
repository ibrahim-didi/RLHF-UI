# src/rlhf_ui/models/trainer.py
"""
Reward model definition and trainer implementation.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, List, Union, Dict, Any

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms # type: ignore
from torchvision.models import resnet50, ResNet50_Weights # type: ignore
from safetensors.torch import save_file, load_file
from PIL import Image  # Add missing PIL Image import

from rlhf_ui.data.dataset import PreferenceDataset, PreferenceDataCollator
from rlhf_ui.visualization import (
    init_wandb,
    log_metrics,
    log_model,
    finish_run
)

logger = logging.getLogger(__name__)

class RewardModel(nn.Module):
    """
    Reward model architecture for RLHF.
    Predicts a scalar reward value for an input image (and optional text).
    """
    
    def __init__(
        self, 
        use_text: bool = False, 
        text_embedding_size: int = 0,
        pretrained: bool = True
    ):
        """
        Initialize the reward model.
        
        Args:
            use_text: Whether to use text inputs along with images
            text_embedding_size: Size of text embeddings if using text
            pretrained: Whether to use pretrained image encoder
        """
        super().__init__()
        
        # Image encoder
        if pretrained:
            self.image_encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.image_encoder = resnet50(weights=None)
            
        # Remove classification head
        self.image_encoder = nn.Sequential(*list(self.image_encoder.children())[:-1])
        
        # Feature dimension from ResNet
        img_feature_dim = 2048
        
        # Determine combined dimension
        self.use_text = use_text
        if use_text and text_embedding_size > 0:
            combined_dim = img_feature_dim + text_embedding_size
        else:
            combined_dim = img_feature_dim
            
        # Reward head: predicts a single scalar reward
        self.reward_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
    
    def forward(
        self, 
        img: torch.Tensor, 
        text_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the reward model.
        
        Args:
            img: Image tensor [batch_size, 3, H, W]
            text_embedding: Text embedding tensor [batch_size, text_dim] (optional)
            
        Returns:
            torch.Tensor: Predicted reward values [batch_size, 1]
        """
        # Process image
        img_features = self.image_encoder(img)
        img_features = img_features.squeeze(-1).squeeze(-1)
        
        # Combine with text if provided
        if self.use_text and text_embedding is not None:
            combined_features = torch.cat([img_features, text_embedding], dim=1)
        else:
            combined_features = img_features
        
        # Predict reward
        reward = self.reward_head(combined_features)
        
        return reward


class RewardModelTrainer:
    """
    Trainer for reward models based on human preferences.
    """
    
    def __init__(
        self,
        preference_file: Union[str, Path],
        image_folder: Union[str, Path],
        model_output_dir: Union[str, Path] = "reward_model",
        device: Optional[torch.device] = None
    ):
        """
        Initialize the reward model trainer.
        
        Args:
            preference_file: Path to CSV file with preference data
            image_folder: Folder containing the images
            model_output_dir: Directory to save model checkpoints
            device: Device to use for training (auto-detected if None)
        """
        self.preference_file = Path(preference_file)
        self.image_folder = Path(image_folder)
        self.model_output_dir = Path(model_output_dir)
        self.model_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load preference data
        self._load_preference_data()
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize tracking variables
        self.wandb_initialized = False
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.best_model_path = None
        
        # Best model tracking
        self.epoch_losses = []
        self.epoch_accs = []
    
    def _load_preference_data(self) -> None:
        """Load preference data from CSV file."""
        if not self.preference_file.exists():
            raise FileNotFoundError(f"Preference file not found: {self.preference_file}")
            
        try:
            self.preferences_df = pd.read_csv(self.preference_file)
            
            # Exclude ties/skips for initial training
            initial_count = len(self.preferences_df)
            self.preferences_df = self.preferences_df[self.preferences_df['preferred'] > 0]
            
            logger.info(f"Loaded {len(self.preferences_df)} preference records from {self.preference_file}.")
            logger.info(f"Excluded {initial_count - len(self.preferences_df)} ties/skips.")
        except Exception as e:
            logger.error(f"Error loading preference data: {e}")
            # Create an empty DataFrame with the expected structure
            self.preferences_df = pd.DataFrame({
                'image1': [],
                'image2': [],
                'preferred': [],
                'prompt': [],
                'timestamp': [],
                'rater_id': [],
                'response_time_ms': []
            })
            logger.warning("Created empty preference DataFrame due to loading error")
    
    def _initialize_model(self, text_embedding_size: int = 0) -> None:
        """
        Initialize the reward model architecture.
        
        Args:
            text_embedding_size: Size of text embeddings if using prompts
        """
        logger.info("Initializing model architecture...")
        
        # Determine if we should use text
        self.use_text = text_embedding_size > 0
        if self.use_text and not self.preferences_df.empty:
            # Check if prompts are available in the data
            has_prompts = (
                'prompt' in self.preferences_df.columns and
                not self.preferences_df['prompt'].isna().all() and 
                len(self.preferences_df['prompt'].str.strip()) > 0
            )
            
            if not has_prompts:
                logger.warning("Text embeddings requested but no prompts found in data. Disabling text.")
                self.use_text = False
                text_embedding_size = 0
        
        # Create model
        self.model = RewardModel(
            use_text=self.use_text,
            text_embedding_size=text_embedding_size,
            pretrained=True
        )
        
        # Text encoder if needed
        if self.use_text:
            logger.info("Initializing text encoder...")
            try:
                from transformers import AutoModel, AutoTokenizer # type: ignore
                
                # Load text encoder
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
                self.text_encoder = AutoModel.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # Freeze text encoder
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
                    
                self.text_encoder.to(self.device)
                
            except ImportError:
                logger.warning("transformers package not found. Disabling text embedding.")
                self.use_text = False
        
        # Move model to device
        self.model.to(self.device)
        logger.info("Model initialized successfully.")
    
    def _create_dataset(self, batch_size: int = 8) -> None:
        """
        Create dataset and dataloader from the preference pairs.
        
        Args:
            batch_size: Batch size for training
        """
        logger.info("Creating dataset and dataloader...")
        
        # Create dataset
        if self.use_text and hasattr(self, 'tokenizer'):
            self.dataset = PreferenceDataset(
                self.preferences_df, 
                self.image_folder, 
                self.transform, 
                self.tokenizer,
                use_text=True
            )
            collate_fn = PreferenceDataCollator(self.tokenizer, use_text=True)
        else:
            self.dataset = PreferenceDataset(
                self.preferences_df, 
                self.image_folder, 
                self.transform
            )
            collate_fn = None
        
        # Create dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            collate_fn=collate_fn
        )
        
        logger.info(f"Dataset created with {len(self.dataset)} samples.")
    
    def train(
        self, 
        epochs: int = 10, 
        lr: float = 1e-4, 
        batch_size: int = 8,
        text_embedding_size: int = 0, 
        progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
        use_wandb: bool = True,
        wandb_project: str = "rlhf-reward-model",
        wandb_run_name: Optional[str] = None
    ) -> str:
        """
        Train the reward model on preference data.
        
        Args:
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Batch size for training
            text_embedding_size: Size of text embeddings if using prompts
            progress_callback: Optional callback for progress updates
            use_wandb: Whether to use Weights & Biases for tracking
            wandb_project: W&B project name
            wandb_run_name: Optional W&B run name (auto-generated if None)
            
        Returns:
            str: Path to the best model checkpoint
        """
        # Initialize model architecture
        self._initialize_model(text_embedding_size)
        
        # Setup dataset and dataloader
        self._create_dataset(batch_size)
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Setup loss function
        # Bradley-Terry loss for preference learning
        loss_fn = nn.BCEWithLogitsLoss()
        
        # Best model tracking
        best_accuracy = 0.0
        best_model_path = None
        
        # Track metrics for UI display
        self.epoch_losses = []
        self.epoch_accs = []
        
        # Setup W&B tracking if enabled
        if use_wandb:
            # Initialize W&B run for this training session
            config = {
                "epochs": epochs,
                "learning_rate": lr,
                "batch_size": batch_size,
                "use_text": text_embedding_size > 0,
                "text_embedding_size": text_embedding_size,
                "device": str(self.device),
                "model_architecture": "ResNet50",
                "dataset_size": len(self.dataset),
                "optimizer": "Adam"
            }
            
            # Generate a run name if not provided
            if wandb_run_name is None:
                wandb_run_name = f"reward-model-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            init_wandb(
                project_name=wandb_project,
                experiment_name=wandb_run_name,
                config=config,
                tags=["reward-model", "training"]
            )
            self.wandb_initialized = True
        
        # Training loop
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            # Loop through batches
            for i, batch in enumerate(self.dataloader):
                # Handle data based on whether we have text
                if isinstance(batch, dict) and 'text' in batch:
                    img1 = batch['img1'].to(self.device)
                    img2 = batch['img2'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Process text
                    text_inputs = batch['text']
                    text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                    
                    with torch.no_grad():
                        text_emb = self.text_encoder(**text_inputs).last_hidden_state.mean(dim=1)
                    
                    # Forward pass for both images
                    r1 = self.model(img1, text_emb).squeeze()
                    r2 = self.model(img2, text_emb).squeeze()
                elif isinstance(batch, dict):
                    img1 = batch['img1'].to(self.device)
                    img2 = batch['img2'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Forward pass
                    r1 = self.model(img1).squeeze()
                    r2 = self.model(img2).squeeze()
                else:
                    # Handle tuple format
                    img1, img2, labels = batch
                    img1, img2 = img1.to(self.device), img2.to(self.device)
                    labels = labels.to(self.device)
                    
                    r1 = self.model(img1).squeeze()
                    r2 = self.model(img2).squeeze()
                
                # Compute preference probabilities (Bradley-Terry model)
                # P(A > B) = sigmoid(r_A - r_B)
                reward_diff = r1 - r2
                
                # Convert preference labels (1, 2) to 0/1 for BCE loss
                # 1 means first image preferred, 2 means second image preferred
                target = (labels == 2).float()
                
                # Make sure reward_diff and target have the same shape
                if reward_diff.dim() != target.dim():
                    if reward_diff.dim() == 1:
                        reward_diff = reward_diff.unsqueeze(1)  # Add channel dimension
                    elif target.dim() == 1:
                        target = target.unsqueeze(1)  # Add channel dimension
                
                # Calculate loss
                loss = loss_fn(reward_diff, target)
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                
                # Calculate accuracy (based on preference prediction)
                pred_probs = torch.sigmoid(reward_diff)
                predicted = (pred_probs >= 0.5).float()
                
                # Ensure target and predicted have the same shape for comparison
                if predicted.shape != target.shape:
                    if predicted.dim() > target.dim():
                        target = target.unsqueeze(-1)
                    elif target.dim() > predicted.dim():
                        predicted = predicted.unsqueeze(-1)
                
                correct += (predicted == target).sum().item()
                total += target.size(0)
                
                # Report progress
                progress_pct = int(100 * (i + 1) / len(self.dataloader))
                
                # Average loss for reporting
                avg_loss = train_loss / (i + 1)
                
                # Accuracy so far
                accuracy = 100 * correct / total if total > 0 else 0
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(epoch, progress_pct, avg_loss, accuracy)
                    
                # Log metrics to W&B
                if use_wandb and i % 10 == 0:  # Log every 10 batches
                    current_step = epoch * len(self.dataloader) + i
                    metrics = {
                        "train/batch_loss": loss.item(),
                        "train/running_loss": avg_loss,
                        "train/running_accuracy": accuracy,
                        "_step": current_step  # Explicitly log step
                    }
                    log_metrics(metrics, step=current_step)
            
            # Compute final metrics for the epoch
            epoch_loss = train_loss / len(self.dataloader)
            epoch_accuracy = 100 * correct / total if total > 0 else 0
            
            # Store metrics for UI display
            self.epoch_losses.append(epoch_loss)
            self.epoch_accs.append(epoch_accuracy)
            
            # Log epoch metrics
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
            
            if use_wandb:
                current_step = (epoch + 1) * len(self.dataloader)
                metrics = {
                    "train/loss": epoch_loss,
                    "train/accuracy": epoch_accuracy,
                    "train/best_accuracy": max(self.epoch_accs),
                    "train/epoch": epoch,
                    "_step": current_step  # Explicitly log step
                }
                log_metrics(metrics, step=current_step)
            
            # Save checkpoint if it's the best model so far
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_loss = epoch_loss
                best_epoch = epoch
                best_model_path = self._save_model(epoch, epoch_loss, epoch_accuracy)
                
                # Store best metrics for access from UI
                self.best_accuracy = best_accuracy
                self.best_loss = best_loss
                self.best_epoch = best_epoch
                
                # Log best model
                if use_wandb:
                    log_metrics({"train/best_accuracy": best_accuracy}, step=current_step)
                    
                    # Log model artifact
                    metadata = {
                        "epoch": epoch,
                        "accuracy": epoch_accuracy,
                        "loss": epoch_loss
                    }
                    log_model(self.model, f"reward-model-epoch-{epoch}", metadata)
        
        # Training completed
        self.best_model_path = best_model_path
        
        # Finalize W&B run
        if use_wandb:
            finish_run()
            self.wandb_initialized = False
        
        logger.info(f"Training completed. Best accuracy: {best_accuracy:.2f}%")
        logger.info(f"Best model saved to: {best_model_path}")
        
        return best_model_path
    
    def _save_model(self, epoch: int, loss: float, accuracy: float) -> str:
        """
        Save the model checkpoint.
        
        Args:
            epoch: Current epoch number
            loss: Training loss value
            accuracy: Training accuracy
            
        Returns:
            str: Path to the saved model
        """
        try:
            # Create model directory
            self.model_output_dir.mkdir(exist_ok=True, parents=True)
            
            # Define checkpoint path
            checkpoint_path = self.model_output_dir / f"reward_model_epoch_{epoch:03d}.safetensors"
            
            # Prepare metadata
            metadata = {
                "epoch": str(epoch),
                "loss": str(loss),
                "accuracy": str(accuracy),
                "timestamp": datetime.now().isoformat(),
                "use_text": str(self.use_text)
            }
            
            # Save model weights using safetensors
            model_state_dict = self.model.state_dict()
            save_file(model_state_dict, checkpoint_path, metadata=metadata)
            
            logger.info(f"Model checkpoint saved to {checkpoint_path}")
            return str(checkpoint_path)
        except Exception as e:
            logger.error(f"Error saving model checkpoint: {e}")
            return ""
    
    def export_for_rlhf(self, output_path: Optional[Path] = None) -> Path:
        """
        Export the trained reward model in a format suitable for RLHF fine-tuning.
        
        Args:
            output_path: Path to save the exported model
            
        Returns:
            Path: Path to the exported model
        """
        logger.info("Exporting reward model for RLHF...")
        
        if output_path is None:
            output_path = self.model_output_dir / "reward_model_for_rlhf.safetensors"
        else:
            output_path = Path(output_path)
            
        # Ensure the output directory exists
        output_path.parent.mkdir(exist_ok=True, parents=True)
            
        # Check for model checkpoints
        checkpoints = list(self.model_output_dir.glob("reward_model_epoch_*.safetensors"))
        if not checkpoints:
            raise ValueError("No checkpoints found. Train the model first.")
        
        # Find the best checkpoint based on metadata
        metadata_files = [
            Path(str(checkpoint).replace('.safetensors', '_metadata.json')) 
            for checkpoint in checkpoints
        ]
        
        best_idx = -1
        best_loss = float('inf')
        
        for i, metadata_file in enumerate(metadata_files):
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        if metadata['loss'] < best_loss:
                            best_loss = metadata['loss']
                            best_idx = i
                except Exception as e:
                    logger.error(f"Error reading metadata file {metadata_file}: {e}")
        
        if best_idx == -1:
            raise ValueError("Could not find valid checkpoint metadata")
            
        # Load the best checkpoint
        best_checkpoint = checkpoints[best_idx]
        logger.info(f"Best checkpoint selected: {best_checkpoint}")
        
        try:
            # Load the checkpoint tensors
            checkpoint_tensors = load_file(str(best_checkpoint))
            
            # Export with standardized format for RLHF
            export_tensors = {}
            for key, value in checkpoint_tensors.items():
                # Convert PyTorch parameter naming to a more standard format
                if key.startswith("image_encoder."):
                    export_key = key.replace("image_encoder.", "image_encoder/")
                    export_tensors[export_key] = value
                elif key.startswith("reward_head."):
                    export_key = key.replace("reward_head.", "reward_head/")
                    export_tensors[export_key] = value
                else:
                    # Keep the key as is if it doesn't match the patterns
                    export_tensors[key] = value
            
            # Save the export file
            save_file(export_tensors, str(output_path))
            
            # Create configuration file
            with open(metadata_files[best_idx], 'r') as f:
                metadata = json.load(f)
            
            config = {
                'image_size': 224,
                'prediction_type': 'reward',
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'format': 'safetensors',
                'source_checkpoint': str(best_checkpoint.name),
                'export_timestamp': datetime.now().isoformat(),
                'accuracy': metadata.get('accuracy', 0),
                'use_text': metadata.get('use_text', False)
            }
            
            # Save the config file
            config_path = output_path.with_suffix('.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            logger.info(f"Exported reward model for RLHF to {output_path}")
            logger.info(f"Exported config to {config_path}")
            
            return output_path
        except Exception as e:
            logger.error(f"Error exporting model: {e}")
            raise
        
    def load_model(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Load a saved model checkpoint.
        
        Args:
            checkpoint_path: Path to the model checkpoint
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Try to load metadata to determine model configuration
        metadata_path = checkpoint_path.with_suffix('').with_suffix('.json')
        if not metadata_path.exists():
            metadata_path = Path(str(checkpoint_path).replace('.safetensors', '_metadata.json'))
        
        use_text = False
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    use_text = metadata.get('use_text', False)
            except Exception as e:
                logger.warning(f"Error reading metadata, defaulting to no text: {e}")
        
        # Initialize model with the right configuration
        text_embedding_size = 768 if use_text else 0
        self._initialize_model(text_embedding_size)
        
        try:
            # Load weights
            tensors = load_file(str(checkpoint_path))
            
            # Convert keys back to PyTorch format if needed
            state_dict = {}
            for key, value in tensors.items():
                if '/' in key:
                    # Convert from export format back to PyTorch format
                    new_key = key.replace('image_encoder/', 'image_encoder.').replace('reward_head/', 'reward_head.')
                    state_dict[new_key] = value
                else:
                    state_dict[key] = value
            
            # Load the state dict
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                logger.warning(f"Missing keys when loading model: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading model: {unexpected_keys}")
                
            self.model.eval()
            
            logger.info(f"Model loaded from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict_reward(self, image_path: Union[str, Path], prompt: Optional[str] = None) -> float:
        """
        Predict the reward value for a single image.
        
        Args:
            image_path: Path to the image
            prompt: Optional text prompt for context
            
        Returns:
            float: Predicted reward value
        """
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Model not initialized. Call train() or load_model() first.")
        
        # Prepare image
        image_path = Path(image_path)
        try:
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
                
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise
        
        # Prepare text if needed
        text_embedding = None
        if self.use_text and prompt and hasattr(self, 'text_encoder') and hasattr(self, 'tokenizer'):
            try:
                encoded_text = self.tokenizer(
                    prompt,
                    padding='max_length',
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    text_embedding = self.text_encoder(**encoded_text).last_hidden_state.mean(dim=1)
            except Exception as e:
                logger.error(f"Error processing text prompt: {e}")
                # Continue without text embedding
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            try:
                reward = self.model(img_tensor, text_embedding).item()
                return reward
            except Exception as e:
                logger.error(f"Error during model inference: {e}")
                raise
    
    def batch_predict(
        self, 
        image_paths: List[Union[str, Path]], 
        prompts: Optional[List[str]] = None
    ) -> List[float]:
        """
        Predict reward values for a batch of images.
        
        Args:
            image_paths: List of paths to images
            prompts: Optional list of text prompts (one per image)
            
        Returns:
            List[float]: Predicted reward values
        """
        if not hasattr(self, 'model') or self.model is None:
            raise ValueError("Model not initialized. Call train() or load_model() first.")
        
        # Validate inputs
        if prompts is not None and len(prompts) != len(image_paths):
            raise ValueError("Number of prompts must match number of images")
        
        # Prepare images
        images = []
        valid_indices = []
        for i, path in enumerate(image_paths):
            try:
                path = Path(path)
                if not path.exists():
                    logger.warning(f"Image not found: {path}, skipping")
                    continue
                    
                img = Image.open(path).convert('RGB')
                img_tensor = self.transform(img)
                images.append(img_tensor)
                valid_indices.append(i)
            except Exception as e:
                logger.error(f"Error loading image {path}: {e}")
                # Skip this image
        
        if not images:
            logger.warning("No valid images to process")
            return []
            
        img_batch = torch.stack(images).to(self.device)
        
        # Prepare text if needed
        text_embedding = None
        if self.use_text and prompts and hasattr(self, 'text_encoder') and hasattr(self, 'tokenizer'):
            try:
                # Filter prompts to match valid images
                valid_prompts = [prompts[i] for i in valid_indices]
                
                encoded_text = self.tokenizer(
                    valid_prompts,
                    padding='max_length',
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                ).to(self.device)
                
                with torch.no_grad():
                    text_embedding = self.text_encoder(**encoded_text).last_hidden_state.mean(dim=1)
            except Exception as e:
                logger.error(f"Error processing text prompts: {e}")
                # Continue without text embedding
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            try:
                rewards = self.model(img_batch, text_embedding).squeeze().cpu().tolist()
                
                # Handle case of single item
                if not isinstance(rewards, list):
                    rewards = [rewards]
                
                return rewards
            except Exception as e:
                logger.error(f"Error during batch inference: {e}")
                return []