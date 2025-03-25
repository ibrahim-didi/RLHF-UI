# src/rlhf_ui/data/sampler.py
"""
Image pair sampling strategies for preference collection.
"""

import random
import logging
import numpy as np
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

logger = logging.getLogger(__name__)

class ImagePairSampler:
    """
    Sampler for selecting pairs of images for human preference collection.
    Implements various sampling strategies.
    """
    
    def __init__(
        self, 
        image_folder: Path,
        preferences_df: pd.DataFrame,
        strategy: str = "random",
        cache_embeddings: bool = True
    ):
        """
        Initialize the image pair sampler.
        
        Args:
            image_folder: Path to folder containing images
            preferences_df: DataFrame of collected preferences
            strategy: Sampling strategy ('random', 'active', 'diversity')
            cache_embeddings: Whether to cache image embeddings for efficiency
        """
        self.image_folder = Path(image_folder)
        self.preferences_df = preferences_df
        self.strategy = strategy
        self.cache_embeddings = cache_embeddings
        
        # Load image paths
        self._load_image_paths()
        
        # Initialize embedding model if needed
        if strategy in ["active", "diversity"]:
            self._initialize_embedding_model()
    
    def _load_image_paths(self) -> None:
        """Load paths to all valid images in the folder."""
        self.image_paths = list(self.image_folder.glob("*.jpg")) + \
                          list(self.image_folder.glob("*.png")) + \
                          list(self.image_folder.glob("*.jpeg"))
        
        if len(self.image_paths) < 2:
            raise ValueError(f"Not enough images found in {self.image_folder}")
            
        logger.info(f"Loaded {len(self.image_paths)} images from {self.image_folder}")
    
    def _initialize_embedding_model(self) -> None:
        """Initialize model for embedding images."""
        logger.info("Initializing image embedding model")
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained ResNet model
        self.embedding_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Use the model without the final classification layer
        self.embedding_model = torch.nn.Sequential(*list(self.embedding_model.children())[:-1])
        self.embedding_model.to(self.device)
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
    
    def _get_image_embedding(self, image_path: Path) -> np.ndarray:
        """
        Get embedding for an image using the model.
        
        Args:
            image_path: Path to the image
            
        Returns:
            np.ndarray: Image embedding vector
        """
        # Check cache first
        image_path_str = str(image_path)
        if self.cache_embeddings and image_path_str in self.embedding_cache:
            return self.embedding_cache[image_path_str]
            
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.embedding_model(img_tensor).squeeze().cpu().numpy()
                
            # Cache result if enabled
            if self.cache_embeddings:
                self.embedding_cache[image_path_str] = embedding
            
            return embedding
        except Exception as e:
            logger.error(f"Error computing embedding for {image_path}: {e}")
            # Return a random embedding as fallback
            return np.random.randn(2048)
    
    def sample_pair(self) -> Tuple[Path, Path]:
        """
        Sample a pair of images according to the selected strategy.
        
        Returns:
            Tuple[Path, Path]: Pair of image paths
        """
        if self.strategy == "random":
            return self._random_sampling()
        elif self.strategy == "active":
            return self._active_sampling()
        elif self.strategy == "diversity":
            return self._diversity_sampling()
        else:
            # Default to random if strategy not recognized
            logger.warning(f"Unknown sampling strategy '{self.strategy}', falling back to random")
            return self._random_sampling()
    
    def _random_sampling(self) -> Tuple[Path, Path]:
        """
        Simple random sampling of image pairs.
        
        Returns:
            Tuple[Path, Path]: Pair of image paths
        """
        # Ensure we get two different images
        img1, img2 = random.sample(self.image_paths, 2)
        return img1, img2
    
    def _active_sampling(self) -> Tuple[Path, Path]:
        """
        Active learning sampling strategy.
        Prioritizes images with fewest comparisons and ensures diversity.
        
        Returns:
            Tuple[Path, Path]: Pair of image paths
        """
        # If we don't have much data yet, fall back to random sampling
        if len(self.preferences_df) < 10:
            return self._random_sampling()
            
        # Count comparisons per image
        comparison_counts = {}
        for _, row in self.preferences_df.iterrows():
            img1 = row['image1']
            img2 = row['image2']
            comparison_counts[img1] = comparison_counts.get(img1, 0) + 1
            comparison_counts[img2] = comparison_counts.get(img2, 0) + 1
            
        # Sort images by comparison count (ascending)
        sorted_images = []
        for img in self.image_paths:
            img_str = str(img)
            count = comparison_counts.get(img_str, 0)
            sorted_images.append((img, count))
        
        sorted_images.sort(key=lambda x: x[1])
        
        # Pick images with low counts (add some randomization)
        candidates = sorted_images[:max(3, len(sorted_images) // 5)]
        img1 = random.choice(candidates)[0]
        
        # For the second image, find a diverse pair
        img1_emb = self._get_image_embedding(img1)
        
        # Get embeddings and distances for other images
        distances = []
        for img, _ in sorted_images:
            if img != img1:
                img_emb = self._get_image_embedding(img)
                dist = np.linalg.norm(img1_emb - img_emb)
                distances.append((img, dist))
        
        # Sample with probability proportional to distance (more diverse pairs)
        distances.sort(key=lambda x: x[1], reverse=True)
        
        # Take one of the top diverse images with some randomization
        diverse_candidates = distances[:max(5, len(distances) // 4)]
        img2 = random.choice(diverse_candidates)[0]
        
        return img1, img2
    
    def _diversity_sampling(self) -> Tuple[Path, Path]:
        """
        Diversity sampling strategy.
        Maximizes coverage of the image space by comparing images 
        that haven't been compared before.
        
        Returns:
            Tuple[Path, Path]: Pair of image paths
        """
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
        
        # Find an image with fewest comparisons
        min_comparisons = float('inf')
        least_compared_img = None
        
        # First check for any images that haven't been compared at all
        uncovered_images = set(str(img) for img in self.image_paths) - set(comparison_matrix.keys())
        if uncovered_images:
            least_compared_img = next((img for img in self.image_paths if str(img) in uncovered_images), None)
        
        # If all images have been compared at least once
        if least_compared_img is None:
            for img in self.image_paths:
                img_str = str(img)
                if img_str in comparison_matrix:
                    num_comparisons = len(comparison_matrix[img_str])
                    if num_comparisons < min_comparisons:
                        min_comparisons = num_comparisons
                        least_compared_img = img
        
        # Find an image that hasn't been compared with the first one
        candidates = []
        for img in self.image_paths:
            if img != least_compared_img:
                img_str = str(img)
                img1_str = str(least_compared_img)
                
                # If they haven't been compared yet
                if (img_str not in comparison_matrix or 
                    img1_str not in comparison_matrix or
                    img_str not in comparison_matrix[img1_str]):
                    candidates.append(img)
        
        if candidates:
            # Pick a random candidate
            return least_compared_img, random.choice(candidates)
        else:
            # All images have been compared with each other, fall back to random
            return self._random_sampling()