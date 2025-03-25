import os
import random
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from safetensors.torch import save_file, load_file
import logging

# Setup logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Define the dataset class outside of the method to make it picklable
class PreferenceDataset(Dataset):
    def __init__(self, df, image_folder, transform, tokenizer=None, use_text=False):
        self.df = df
        self.image_folder = image_folder
        self.transform = transform
        self.tokenizer = tokenizer
        self.use_text = use_text
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load images
        img1_path = row['image1']
        img2_path = row['image2']
        
        # Handle relative vs absolute paths
        if not os.path.isabs(img1_path):
            img1_path = os.path.join(self.image_folder, os.path.basename(img1_path))
        if not os.path.isabs(img2_path):
            img2_path = os.path.join(self.image_folder, os.path.basename(img2_path))
        
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        img1_tensor = self.transform(img1)
        img2_tensor = self.transform(img2)
        
        # Create label (1 if img1 is preferred, 0 if img2 is preferred)
        label = 1.0 if row['preferred'] == 1 else 0.0
        
        # Process text if available
        if self.use_text and self.tokenizer is not None and row['prompt']:
            text = row['prompt']
            encoded_text = self.tokenizer(
                text, 
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            return img1_tensor, img2_tensor, label, encoded_text
        
        return img1_tensor, img2_tensor, label


class PreferenceDataCollector:
    """
    A system for collecting human preferences between pairs of images.
    Implements best practices for RLHF data collection.
    """
    
    def __init__(self, image_folder, output_folder="preference_data", 
                 window_size=(1200, 700), sampling_strategy="random"):
        """
        Initialize the preference data collector.
        
        Args:
            image_folder (str): Path to folder containing images
            output_folder (str): Path to save preference data
            window_size (tuple): Size of the UI window
            sampling_strategy (str): Strategy for sampling image pairs
                                    ("random", "active", "diversity")
        """
        self.image_folder = Path(image_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True, parents=True)
        
        # Load image paths
        self.image_paths = list(self.image_folder.glob("*.jpg")) + \
                           list(self.image_folder.glob("*.png")) + \
                           list(self.image_folder.glob("*.jpeg"))
        
        if len(self.image_paths) < 2:
            raise ValueError(f"Not enough images found in {image_folder}")
            
        # Setup preference database
        self.preference_file = self.output_folder / "preferences.csv"
        if not self.preference_file.exists():
            self._initialize_preference_database()
        else:
            self.preferences_df = pd.read_csv(self.preference_file)
        
        # Setup UI elements
        self.window_size = window_size
        self.current_pair = None
        self.prompt = ""  # Optional prompt for context
        
        # Sampling strategy
        self.sampling_strategy = sampling_strategy
        if sampling_strategy == "active":
            # For active sampling, we'll need an embedding model
            self._initialize_embedding_model()
        
        # Statistics
        self.total_comparisons = len(self.preferences_df) if hasattr(self, 'preferences_df') else 0
        self.session_completed = False
        
    def _initialize_preference_database(self):
        """Create an empty preference database."""
        self.preferences_df = pd.DataFrame({
            'image1': [],
            'image2': [],
            'preferred': [],  # 1 for image1, 2 for image2, 0 for tie/skip
            'prompt': [],
            'timestamp': [],
            'rater_id': [],
            'response_time_ms': []
        })
        self.preferences_df.to_csv(self.preference_file, index=False)
        
    def _initialize_embedding_model(self):
        """Initialize model for embedding images for active learning."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained ResNet model
        self.embedding_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Use the model without the final classification layer
        self.embedding_model = nn.Sequential(*list(self.embedding_model.children())[:-1])
        self.embedding_model.to(device)
        self.embedding_model.eval()
        
        # Setup image transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Cache for image embeddings
        self.embedding_cache = {}
        
    def _get_image_embedding(self, image_path):
        """Get embedding for an image using the model."""
        if str(image_path) in self.embedding_cache:
            return self.embedding_cache[str(image_path)]
            
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0)
        
        device = next(self.embedding_model.parameters()).device
        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            embedding = self.embedding_model(img_tensor).squeeze().cpu().numpy()
            
        self.embedding_cache[str(image_path)] = embedding
        return embedding
        
    def sample_image_pair(self):
        """
        Sample a pair of images for comparison based on the selected strategy.
        
        Returns:
            tuple: (image_path1, image_path2)
        """
        if self.sampling_strategy == "random":
            # Simple random sampling
            img1, img2 = random.sample(self.image_paths, 2)
            return img1, img2
            
        elif self.sampling_strategy == "active":
            # Active learning strategy - sample pairs with highest uncertainty
            if len(self.preferences_df) < 10:
                # Not enough data for active learning yet, fall back to random
                return random.sample(self.image_paths, 2)
                
            # Simple approach: find images with fewest comparisons
            comparison_counts = {}
            for _, row in self.preferences_df.iterrows():
                comparison_counts[row['image1']] = comparison_counts.get(row['image1'], 0) + 1
                comparison_counts[row['image2']] = comparison_counts.get(row['image2'], 0) + 1
                
            # Sort images by comparison count (ascending)
            sorted_images = sorted(
                [(img, comparison_counts.get(str(img), 0)) for img in self.image_paths],
                key=lambda x: x[1]
            )
            
            # Pick the least compared image
            img1 = sorted_images[0][0]
            
            # Find a diverse second image (using embeddings)
            img1_emb = self._get_image_embedding(img1)
            
            # Get embeddings for other images
            distances = []
            for img in self.image_paths:
                if img != img1:
                    img_emb = self._get_image_embedding(img)
                    dist = np.linalg.norm(img1_emb - img_emb)
                    distances.append((img, dist))
            
            # Sample with probability proportional to distance (more diverse pairs)
            distances.sort(key=lambda x: x[1], reverse=True)
            img2 = distances[0][0]  # Most different image
            
            return img1, img2
            
        elif self.sampling_strategy == "diversity":
            # Diversity sampling - maximize coverage of the image space
            # This is a simplified version focusing on comparing different images
            
            # Track which images have been compared with which
            comparison_matrix = {}
            for _, row in self.preferences_df.iterrows():
                img1, img2 = row['image1'], row['image2']
                if img1 not in comparison_matrix:
                    comparison_matrix[img1] = set()
                if img2 not in comparison_matrix:
                    comparison_matrix[img2] = set()
                comparison_matrix[img1].add(img2)
                comparison_matrix[img2].add(img1)
            
            # Find the image with fewest comparisons
            min_comparisons = float('inf')
            least_compared_img = None
            
            for img in self.image_paths:
                img_str = str(img)
                if img_str not in comparison_matrix:
                    # This image hasn't been compared yet
                    least_compared_img = img
                    break
                    
                num_comparisons = len(comparison_matrix[img_str])
                if num_comparisons < min_comparisons:
                    min_comparisons = num_comparisons
                    least_compared_img = img
            
            # If all images have been compared exhaustively, fall back to random
            if least_compared_img is None:
                return random.sample(self.image_paths, 2)
                
            # Find an image that hasn't been compared with the first one
            candidates = []
            for img in self.image_paths:
                img_str = str(img)
                if img != least_compared_img:
                    if img_str not in comparison_matrix or \
                       str(least_compared_img) not in comparison_matrix[img_str]:
                        candidates.append(img)
            
            if candidates:
                return least_compared_img, random.choice(candidates)
            else:
                # All images have been compared with each other, fall back to random
                other_img = random.choice([img for img in self.image_paths if img != least_compared_img])
                return least_compared_img, other_img
                
        # Default to random if strategy is not recognized
        return random.sample(self.image_paths, 2)
        
    def record_preference(self, image1, image2, preferred, prompt="", 
                         rater_id="default", response_time_ms=None):
        """
        Record a human preference between two images.
        
        Args:
            image1 (Path): Path to first image
            image2 (Path): Path to second image
            preferred (int): 1 if image1 is preferred, 2 if image2, 0 if tie/skip
            prompt (str): Optional context prompt
            rater_id (str): ID of the human rater
            response_time_ms (float): Response time in milliseconds
        """
        new_row = pd.DataFrame({
            'image1': [str(image1)],
            'image2': [str(image2)],
            'preferred': [preferred],
            'prompt': [prompt],
            'timestamp': [datetime.now().isoformat()],
            'rater_id': [rater_id],
            'response_time_ms': [response_time_ms]
        })
        
        self.preferences_df = pd.concat([self.preferences_df, new_row], ignore_index=True)
        self.preferences_df.to_csv(self.preference_file, index=False)
        self.total_comparisons += 1
        
    def start_ui(self):
        """Launch the UI for collecting preferences."""
        self.root = tk.Tk()
        self.root.title("Human Preference Collection for RLHF")
        self.root.geometry(f"{self.window_size[0]}x{self.window_size[1]}")
        
        # Style configurations
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 12))
        style.configure("TLabel", font=("Arial", 12))
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Instructions
        instructions = ttk.Label(main_frame, 
                                text="Which image better matches the prompt? If no prompt, which is better quality?",
                                font=("Arial", 14, "bold"))
        instructions.pack(pady=10)
        
        # Optional prompt display
        self.prompt_label = ttk.Label(main_frame, text="", font=("Arial", 12))
        self.prompt_label.pack(pady=5)
        
        # Images frame
        images_frame = ttk.Frame(main_frame)
        images_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left image
        left_frame = ttk.Frame(images_frame, padding=5)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.left_image_label = ttk.Label(left_frame)
        self.left_image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        left_button = ttk.Button(left_frame, text="Prefer Left (A)", 
                                command=lambda: self._on_preference(1))
        left_button.pack(pady=10)
        
        # Right image
        right_frame = ttk.Frame(images_frame, padding=5)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.right_image_label = ttk.Label(right_frame)
        self.right_image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        right_button = ttk.Button(right_frame, text="Prefer Right (B)", 
                                 command=lambda: self._on_preference(2))
        right_button.pack(pady=10)
        
        # Bottom controls
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=10)
        
        tie_button = ttk.Button(controls_frame, text="Equal / Can't Decide", 
                               command=lambda: self._on_preference(0))
        tie_button.pack(side=tk.LEFT, padx=10)
        
        skip_button = ttk.Button(controls_frame, text="Skip Pair", 
                                command=self._load_next_pair)
        skip_button.pack(side=tk.LEFT, padx=10)
        
        # Add finish button to properly close and proceed to training
        finish_button = ttk.Button(controls_frame, text="Finish Rating", 
                                 command=self._finish_rating,
                                 style="Accent.TButton")
        finish_button.pack(side=tk.RIGHT, padx=10)
        
        # Create a distinct style for the finish button
        style.configure("Accent.TButton", background="#4CAF50", foreground="white", font=("Arial", 12, "bold"))
        
        # Statistics label
        self.stats_label = ttk.Label(main_frame, 
                                    text=f"Total comparisons: {self.total_comparisons}")
        self.stats_label.pack(pady=5)
        
        # Keyboard shortcuts
        self.root.bind('a', lambda e: self._on_preference(1))
        self.root.bind('b', lambda e: self._on_preference(2))
        self.root.bind('e', lambda e: self._on_preference(0))
        self.root.bind('s', lambda e: self._load_next_pair())
        self.root.bind('f', lambda e: self._finish_rating())
        
        # Handle window close event
        self.root.protocol("WM_DELETE_WINDOW", self._finish_rating)
        
        # Load initial pair
        self._load_next_pair()
        
        # Start UI loop
        self.comparison_start_time = datetime.now()
        self.root.mainloop()
        
    def _finish_rating(self):
        """Properly finish the rating session and close the UI."""
        if hasattr(self, 'root') and self.root:
            print(f"Rating session completed with {self.total_comparisons} comparisons.")
            self.session_completed = True
            # Quit the mainloop first, then schedule destruction
            self.root.quit()
            self.root.after_idle(self.root.destroy)
        
    def _on_preference(self, preference):
        """
        Handle preference selection (1 for left, 2 for right, 0 for tie).
        """
        # Do nothing if the session is completed
        if getattr(self, 'session_completed', False):
            return

        if self.current_pair:
            response_time_ms = (datetime.now() - self.comparison_start_time).total_seconds() * 1000
            self.record_preference(
                self.current_pair[0], 
                self.current_pair[1], 
                preference,
                prompt=self.prompt,
                response_time_ms=response_time_ms
            )
            self._load_next_pair()
            
    def _load_next_pair(self):
        """
        Load the next pair of images for comparison.
        """
        # Do not load new pairs if session is finished
        if getattr(self, 'session_completed', False):
            return
        
        # Sample a new pair
        img1, img2 = self.sample_image_pair()
        self.current_pair = (img1, img2)
        
        # Optional: set a prompt (could be based on filenames or metadata)
        self.prompt = ""
        self.prompt_label.config(text=self.prompt if self.prompt else "(No specific prompt)")
        
        # Load and display images
        left_img = Image.open(img1)
        right_img = Image.open(img2)
        
        # Resize while maintaining aspect ratio
        left_img = self._resize_image(left_img)
        right_img = self._resize_image(right_img)
        
        # Convert to PhotoImage for tkinter
        left_tk = ImageTk.PhotoImage(left_img)
        right_tk = ImageTk.PhotoImage(right_img)
        
        # Update labels and keep a reference
        self.left_image_label.config(image=left_tk)
        self.left_image_label.image = left_tk
        
        self.right_image_label.config(image=right_tk)
        self.right_image_label.image = right_tk
        
        # Update stats
        self.stats_label.config(text=f"Total comparisons: {self.total_comparisons}")
        
        # Reset timer
        self.comparison_start_time = datetime.now()
        
    def _resize_image(self, img, max_height=500, max_width=500):
        """Resize image while maintaining aspect ratio."""
        width, height = img.size
        ratio = min(max_width / width, max_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        return img.resize((new_width, new_height), Image.LANCZOS)


class RewardModelTrainer:
    """
    Trains a reward model based on collected human preferences.
    Added logging to track progress throughout the training process.
    """
    
    def __init__(self, preference_file, image_folder, model_output_dir="reward_model"):
        """
        Initialize the reward model trainer.
        
        Args:
            preference_file (str): Path to the CSV file with preference data
            image_folder (str): Folder containing the images
            model_output_dir (str): Directory to save model checkpoints
        """
        logging.info("Initializing RewardModelTrainer...")
        self.preference_file = Path(preference_file)
        self.image_folder = Path(image_folder)
        self.model_output_dir = Path(model_output_dir)
        self.model_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load preference data
        self.preferences_df = pd.read_csv(self.preference_file)
        logging.info(f"Loaded {len(self.preferences_df)} preference records from {self.preference_file}.")
        
        # Exclude ties/skips for initial training
        initial_count = len(self.preferences_df)
        self.preferences_df = self.preferences_df[self.preferences_df['preferred'] > 0]
        logging.info(f"Excluding ties/skips: {len(self.preferences_df)} records remain from {initial_count}.")
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
    def _initialize_model(self, text_embedding_size=0):
        """
        Initialize the reward model architecture.
        
        Args:
            text_embedding_size (int): Size of text embeddings if using prompts
        """
        logging.info("Initializing model architecture...")
        # Base image encoder from ResNet
        base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Remove classification head
        modules = list(base_model.children())[:-1]
        self.image_encoder = nn.Sequential(*modules)
        
        # Feature dimension from ResNet
        img_feature_dim = 2048
        
        # Text encoder if needed
        self.use_text = text_embedding_size > 0
        if self.use_text:
            logging.info("Using text encoder for model.")
            self.text_encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            # Freeze text encoder
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            combined_dim = img_feature_dim + text_embedding_size
        else:
            logging.info("No text encoder used.")
            combined_dim = img_feature_dim
            
        # Reward head: predicts a single scalar reward
        self.reward_head = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
        # Move models to device
        self.image_encoder.to(self.device)
        if self.use_text:
            self.text_encoder.to(self.device)
        self.reward_head.to(self.device)
        logging.info("Model architecture initialized successfully.")
        
    def _create_dataset(self):
        """
        Create a dataset from the preference pairs.
        """
        logging.info("Creating dataset from preference records...")
        # Check if we have prompts
        has_prompts = not self.preferences_df['prompt'].isna().all() and \
                      len(self.preferences_df['prompt'].str.strip()) > 0
        
        if has_prompts and hasattr(self, 'tokenizer') and self.tokenizer is not None:
            self.dataset = PreferenceDataset(
                self.preferences_df, 
                self.image_folder, 
                self.transform, 
                self.tokenizer,
                use_text=True
            )
        else:
            self.dataset = PreferenceDataset(
                self.preferences_df, 
                self.image_folder, 
                self.transform
            )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        logging.info(f"Dataset created with {len(self.dataset)} samples.")
        
    def train(self, epochs=10, lr=1e-4, text_embedding_size=0):
        """
        Train the reward model using the Bradley-Terry model for preferences.
        
        Args:
            epochs (int): Number of training epochs
            lr (float): Learning rate
            text_embedding_size (int): Size of text embeddings if using prompts
        """
        logging.info(f"Starting training: {epochs} epochs, learning rate = {lr}.")
        self._initialize_model(text_embedding_size)
        self._create_dataset()
        
        # Define optimizer
        parameters = list(self.image_encoder.parameters()) + list(self.reward_head.parameters())
        optimizer = optim.Adam(parameters, lr=lr)
        
        # Define scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            logging.info(f"Epoch {epoch+1}/{epochs} started.")
            self.image_encoder.train()
            self.reward_head.train()
            
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in self.dataloader:
                if len(batch) == 3:
                    img1, img2, labels = batch
                    img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
                    img1_emb = self.image_encoder(img1).squeeze(-1).squeeze(-1)
                    img2_emb = self.image_encoder(img2).squeeze(-1).squeeze(-1)
                    r1 = self.reward_head(img1_emb).squeeze()
                    r2 = self.reward_head(img2_emb).squeeze()
                else:  # With text embeddings
                    img1, img2, labels, text_data = batch
                    img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
                    text_inputs = {k: v.squeeze(0).to(self.device) for k, v in text_data.items()}
                    text_emb = self.text_encoder(**text_inputs).last_hidden_state.mean(dim=1)
                    img1_emb = self.image_encoder(img1).squeeze(-1).squeeze(-1)
                    img2_emb = self.image_encoder(img2).squeeze(-1).squeeze(-1)
                    img1_text_emb = torch.cat([img1_emb, text_emb], dim=1)
                    img2_text_emb = torch.cat([img2_emb, text_emb], dim=1)
                    r1 = self.reward_head(img1_text_emb).squeeze()
                    r2 = self.reward_head(img2_text_emb).squeeze()
                
                logits = r1 - r2
                loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                predictions = (logits > 0).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
            
            avg_loss = total_loss / len(self.dataloader)
            accuracy = correct / total
            logging.info(f"Epoch {epoch+1} complete: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
            
            scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                self._save_model(epoch, avg_loss, accuracy)
                
        logging.info("Training completed!")
        
    def _save_model(self, epoch, loss, accuracy):
        """Save the model checkpoint using SafeTensors format."""
        logging.info(f"Saving model checkpoint for epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
        tensor_dict = {}
        for key, value in self.image_encoder.state_dict().items():
            tensor_dict[f"image_encoder.{key}"] = value
        for key, value in self.reward_head.state_dict().items():
            tensor_dict[f"reward_head.{key}"] = value
        if hasattr(self, 'text_encoder') and self.text_encoder is not None:
            for key, value in self.text_encoder.state_dict().items():
                tensor_dict[f"text_encoder.{key}"] = value
        
        save_file(tensor_dict, str(self.model_output_dir / f"reward_model_epoch_{epoch}.safetensors"))
        
        metadata = {
            'epoch': epoch,
            'loss': float(loss),
            'accuracy': float(accuracy),
            'timestamp': datetime.now().isoformat(),
            'model_type': 'reward_model',
            'framework': 'pytorch',
            'format_version': '1.0'
        }
        
        with open(self.model_output_dir / f"reward_model_epoch_{epoch}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        with open(self.model_output_dir / 'config.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        logging.info("Model checkpoint saved.")
            
    def export_for_rlhf(self, output_path=None):
        """
        Export the trained reward model in SafeTensors format suitable for RLHF fine-tuning.
        """
        logging.info("Exporting reward model for RLHF...")
        if output_path is None:
            output_path = self.model_output_dir / "reward_model_for_rlhf.safetensors"
            
        checkpoints = list(self.model_output_dir.glob("reward_model_epoch_*.safetensors"))
        if not checkpoints:
            raise ValueError("No checkpoints found. Train the model first.")
        
        metadata_files = [Path(str(checkpoint).replace('.safetensors', '_metadata.json')) 
                         for checkpoint in checkpoints]
        best_idx = -1
        best_loss = float('inf')
        
        for i, metadata_file in enumerate(metadata_files):
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    if metadata['loss'] < best_loss:
                        best_loss = metadata['loss']
                        best_idx = i
        
        if best_idx == -1:
            raise ValueError("Could not find valid checkpoint metadata")
            
        best_checkpoint = checkpoints[best_idx]
        logging.info(f"Best checkpoint selected: {best_checkpoint}")
        checkpoint_tensors = load_file(str(best_checkpoint))
        
        export_tensors = {}
        for key, value in checkpoint_tensors.items():
            if key.startswith("image_encoder."):
                export_key = key.replace("image_encoder.", "image_encoder/")
                export_tensors[export_key] = value
            elif key.startswith("reward_head."):
                export_key = key.replace("reward_head.", "reward_head/")
                export_tensors[export_key] = value
            elif key.startswith("text_encoder."):
                export_key = key.replace("text_encoder.", "text_encoder/")
                export_tensors[export_key] = value
                
        save_file(export_tensors, str(output_path))
        
        config = {
            'image_size': 224,
            'prediction_type': 'reward',
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'format': 'safetensors',
            'source_checkpoint': str(best_checkpoint.name),
            'export_timestamp': datetime.now().isoformat()
        }
        
        has_text_encoder = any(key.startswith("text_encoder.") for key in checkpoint_tensors)
        if has_text_encoder:
            config['use_text'] = True
            
        config_path = output_path.with_suffix('.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        logging.info(f"Exported reward model for RLHF to {output_path}")
        logging.info(f"Exported config to {config_path}")
        return output_path


# Example usage
if __name__ == "__main__":
    # 1. Collect preferences
    collector = PreferenceDataCollector(
        image_folder="./images",
        sampling_strategy="active"  # Use active learning for efficient data collection
    )
    collector.start_ui()
    
    # 2. Train reward model
    trainer = RewardModelTrainer(
        preference_file="./preference_data/preferences.csv",
        image_folder="./images"
    )
    
    # Use no multiprocessing for compatibility
    trainer.train(epochs=20)
    
    # 3. Export for RLHF
    trainer.export_for_rlhf()