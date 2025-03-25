# src/rlhf_ui/models/embedding.py
"""
Embedding models for images and text used in RLHF.
Provides standardized interfaces for feature extraction.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union, Tuple

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

logger = logging.getLogger(__name__)

class ImageEmbeddingModel:
    """
    Image embedding model that extracts features from images.
    Uses a pre-trained ResNet model by default.
    """
    
    def __init__(
        self, 
        model_name: str = "resnet50",
        pretrained: bool = True,
        device: Optional[torch.device] = None,
        cache_embeddings: bool = True
    ):
        """
        Initialize the image embedding model.
        
        Args:
            model_name: Name of the backbone model to use
            pretrained: Whether to use pre-trained weights
            device: Device to run the model on (defaults to CUDA if available)
            cache_embeddings: Whether to cache embeddings for faster repeated access
        """
        self.model_name = model_name
        self.cache_embeddings = cache_embeddings
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device} for image embedding model")
        
        # Initialize model
        self._initialize_model(model_name, pretrained)
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Initialize embedding cache
        self.embedding_cache = {}
    
    def _initialize_model(self, model_name: str, pretrained: bool) -> None:
        """
        Initialize the backbone model.
        
        Args:
            model_name: Name of the model to use
            pretrained: Whether to use pre-trained weights
        """
        if model_name == "resnet50":
            if pretrained:
                # Use pretrained model
                base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            else:
                base_model = resnet50(weights=None)
                
            # Remove the classification layer
            modules = list(base_model.children())[:-1]
            self.model = nn.Sequential(*modules)
            self.output_dim = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Initialized {model_name} embedding model (pretrained: {pretrained})")
    
    def embed_image(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """
        Generate embedding for a single image.
        
        Args:
            image: Image to embed (file path or PIL Image)
            
        Returns:
            np.ndarray: Image embedding vector
        """
        # Check cache if we have a file path
        if self.cache_embeddings and isinstance(image, (str, Path)):
            cache_key = str(image)
            if cache_key in self.embedding_cache:
                return self.embedding_cache[cache_key]
        
        # Load image if needed
        if isinstance(image, (str, Path)):
            try:
                img = Image.open(image).convert('RGB')
            except Exception as e:
                logger.error(f"Error loading image {image}: {e}")
                raise
        else:
            img = image
        
        # Preprocess image
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            embedding = self.model(img_tensor).squeeze().cpu().numpy()
        
        # Cache result if applicable
        if self.cache_embeddings and isinstance(image, (str, Path)):
            self.embedding_cache[str(image)] = embedding
        
        return embedding
    
    def embed_batch(self, images: List[Union[str, Path, Image.Image]]) -> np.ndarray:
        """
        Generate embeddings for a batch of images.
        
        Args:
            images: List of images to embed
            
        Returns:
            np.ndarray: Batch of image embedding vectors [batch_size, output_dim]
        """
        # Process one by one for now to allow caching and error handling
        embeddings = []
        for img in images:
            try:
                emb = self.embed_image(img)
                embeddings.append(emb)
            except Exception as e:
                logger.error(f"Error embedding image {img}: {e}")
                # Use zeros as fallback for errors
                embeddings.append(np.zeros(self.output_dim))
        
        return np.stack(embeddings)
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.embedding_cache.clear()
        logger.info("Cleared image embedding cache")
    
    def get_similarity(self, img1: Union[str, Path, Image.Image], 
                       img2: Union[str, Path, Image.Image]) -> float:
        """
        Calculate cosine similarity between two images.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            float: Cosine similarity score (-1 to 1)
        """
        emb1 = self.embed_image(img1)
        emb2 = self.embed_image(img2)
        
        # Compute cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def get_nearest_neighbors(
        self, 
        query_image: Union[str, Path, Image.Image],
        image_list: List[Union[str, Path, Image.Image]],
        k: int = 5
    ) -> List[Tuple[Union[str, Path, Image.Image], float]]:
        """
        Find k-nearest neighbors to a query image.
        
        Args:
            query_image: Query image
            image_list: List of candidate images
            k: Number of neighbors to return
            
        Returns:
            List[Tuple[image, similarity]]: K-nearest neighbors with similarity scores
        """
        query_emb = self.embed_image(query_image)
        
        # Compute embeddings for all candidates
        similarities = []
        for img in image_list:
            if img == query_image:
                continue
                
            img_emb = self.embed_image(img)
            sim = np.dot(query_emb, img_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(img_emb))
            similarities.append((img, float(sim)))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return similarities[:k]


class TextEmbeddingModel:
    """
    Text embedding model for encoding prompts and context.
    Uses a pre-trained transformer model.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[torch.device] = None,
        cache_size: int = 1000
    ):
        """
        Initialize the text embedding model.
        
        Args:
            model_name: Name or path of the pre-trained model
            device: Device to run the model on
            cache_size: Maximum number of entries to keep in cache
        """
        self.model_name = model_name
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Initialize cache
        self.cache = {}
        self.cache_size = cache_size
        
        # Load model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the text embedding model and tokenizer."""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            self.output_dim = self.model.config.hidden_size
            logger.info(f"Loaded text embedding model: {self.model_name} (dim={self.output_dim})")
            
        except ImportError:
            logger.error("Failed to import transformers library. Text embedding not available.")
            self.model = None
            self.tokenizer = None
            self.output_dim = 0
    
    def is_available(self) -> bool:
        """Check if the text embedding model is available."""
        return self.model is not None and self.tokenizer is not None
    
    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for a text.
        
        Args:
            text: Text to embed
            
        Returns:
            Optional[np.ndarray]: Text embedding vector or None if model not available
        """
        if not self.is_available():
            logger.warning("Text embedding model not available")
            return None
        
        # Check cache
        if text in self.cache:
            return self.cache[text]
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Use mean of last hidden state as embedding
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        
        # Update cache
        if len(self.cache) >= self.cache_size:
            # Remove a random item if cache is full
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[text] = embedding
        
        return embedding
    
    def embed_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Optional[np.ndarray]: Batch of text embedding vectors or None if model not available
        """
        if not self.is_available():
            logger.warning("Text embedding model not available")
            return None
        
        # Process texts one by one to leverage caching
        embeddings = []
        for text in texts:
            emb = self.embed_text(text)
            embeddings.append(emb)
        
        return np.stack(embeddings)
    
    def get_similarity(self, text1: str, text2: str) -> Optional[float]:
        """
        Calculate cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Optional[float]: Cosine similarity score (-1 to 1) or None if model not available
        """
        if not self.is_available():
            return None
            
        emb1 = self.embed_text(text1)
        emb2 = self.embed_text(text2)
        
        # Compute cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.cache.clear()
        logger.info("Cleared text embedding cache")


class MultiModalEmbedding:
    """
    Combined image and text embedding model for RLHF.
    Allows creating joint embeddings from image-text pairs.
    """
    
    def __init__(
        self,
        image_model: Optional[ImageEmbeddingModel] = None,
        text_model: Optional[TextEmbeddingModel] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the multimodal embedding model.
        
        Args:
            image_model: Image embedding model (created if None)
            text_model: Text embedding model (created if None)
            device: Device to use for computation
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Initialize models if not provided
        if image_model is None:
            self.image_model = ImageEmbeddingModel(device=self.device)
        else:
            self.image_model = image_model
            
        if text_model is None:
            self.text_model = TextEmbeddingModel(device=self.device)
        else:
            self.text_model = text_model
    
    def embed(
        self, 
        image: Union[str, Path, Image.Image],
        text: Optional[str] = None,
        fusion: str = "concat"
    ) -> np.ndarray:
        """
        Generate a joint embedding for an image-text pair.
        
        Args:
            image: Image to embed
            text: Optional text to embed
            fusion: Fusion method ('concat', 'add', 'mult')
            
        Returns:
            np.ndarray: Joint embedding vector
        """
        # Get image embedding
        img_emb = self.image_model.embed_image(image)
        
        # If no text or text model unavailable, return image embedding
        if text is None or not self.text_model.is_available():
            return img_emb
        
        # Get text embedding
        text_emb = self.text_model.embed_text(text)
        
        # Fuse embeddings
        if fusion == "concat":
            # Concatenate embeddings
            return np.concatenate([img_emb, text_emb])
        elif fusion == "add":
            # Ensure dimensions match
            if img_emb.shape[0] != text_emb.shape[0]:
                logger.warning("Cannot add embeddings with different dimensions")
                return img_emb
            # Element-wise addition
            return img_emb + text_emb
        elif fusion == "mult":
            # Ensure dimensions match
            if img_emb.shape[0] != text_emb.shape[0]:
                logger.warning("Cannot multiply embeddings with different dimensions")
                return img_emb
            # Element-wise multiplication
            return img_emb * text_emb
        else:
            logger.warning(f"Unknown fusion method: {fusion}")