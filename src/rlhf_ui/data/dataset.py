# src/rlhf_ui/data/dataset.py
"""
Dataset classes for RLHF model training.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Dict, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

class PreferenceDataset(Dataset):
    """
    Dataset for training reward models from human preference data.
    """
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        image_folder: Union[str, Path], 
        transform=None, 
        tokenizer=None, 
        use_text: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            df: DataFrame containing preference data
            image_folder: Folder containing images
            transform: Image transforms to apply
            tokenizer: Tokenizer for text prompts (if use_text=True)
            use_text: Whether to include text prompts in the dataset
        """
        self.df = df
        self.image_folder = Path(image_folder)
        
        # Setup transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
        
        # Text processing
        self.tokenizer = tokenizer
        self.use_text = use_text and tokenizer is not None
        
        # Exclude ties/skips (where preferred=0)
        self.df = self.df[self.df['preferred'] > 0].reset_index(drop=True)
        
        logger.info(f"Created dataset with {len(self.df)} preference pairs")
    
    def __len__(self) -> int:
        """Get the number of preference pairs."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor, float], 
                                          Tuple[torch.Tensor, torch.Tensor, float, Dict[str, torch.Tensor]]]:
        """
        Get a preference pair item.
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple containing:
                - img1_tensor: First image tensor
                - img2_tensor: Second image tensor
                - label: Preference label (1 if img1 preferred, 0 if img2 preferred)
                - encoded_text: Tokenized text (optional, if use_text=True)
        """
        row = self.df.iloc[idx]
        
        # Load images
        img1_path = row['image1']
        img2_path = row['image2']
        
        # Handle relative vs absolute paths
        if not os.path.isabs(img1_path):
            img1_path = os.path.join(self.image_folder, os.path.basename(img1_path))
        if not os.path.isabs(img2_path):
            img2_path = os.path.join(self.image_folder, os.path.basename(img2_path))
        
        try:
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading images at index {idx}: {e}")
            # Create dummy images as fallback
            img1 = Image.new('RGB', (224, 224), color=(0, 0, 0))
            img2 = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        img1_tensor = self.transform(img1)
        img2_tensor = self.transform(img2)
        
        # Create label (1 if img1 is preferred, 0 if img2 is preferred)
        label = 1.0 if row['preferred'] == 1 else 0.0
        
        # Process text if available and requested
        if self.use_text and row['prompt'] and not pd.isna(row['prompt']):
            text = str(row['prompt'])
            encoded_text = self.tokenizer(
                text, 
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            # Remove batch dimension from tokenizer output
            encoded_text = {k: v.squeeze(0) for k, v in encoded_text.items()}
            
            return img1_tensor, img2_tensor, label, encoded_text
        
        return img1_tensor, img2_tensor, label


class PreferenceDataCollator:
    """
    Data collator for preference pairs with optional text.
    Handles batching of image pairs with text when needed.
    """
    
    def __init__(self, tokenizer=None, use_text: bool = False):
        """
        Initialize the collator.
        
        Args:
            tokenizer: Tokenizer for text prompts
            use_text: Whether the dataset includes text prompts
        """
        self.tokenizer = tokenizer
        self.use_text = use_text and tokenizer is not None
    
    def __call__(self, batch: list) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of preference pairs.
        
        Args:
            batch: List of preference pair items
            
        Returns:
            Dictionary containing batched tensors
        """
        if not batch:
            return {}
            
        # Check if we have text data
        has_text = len(batch[0]) > 3
        
        if has_text:
            img1_tensors, img2_tensors, labels, text_data = [], [], [], {}
            
            for img1, img2, label, text in batch:
                img1_tensors.append(img1)
                img2_tensors.append(img2)
                labels.append(label)
                
                # Initialize text_data dict
                if not text_data:
                    text_data = {k: [] for k in text.keys()}
                
                # Add each text item
                for k, v in text.items():
                    text_data[k].append(v)
            
            # Stack tensors
            img1_batch = torch.stack(img1_tensors)
            img2_batch = torch.stack(img2_tensors)
            labels_batch = torch.tensor(labels)
            
            # Process text data
            text_batch = {}
            for k, v in text_data.items():
                text_batch[k] = torch.stack(v)
            
            return {
                'img1': img1_batch,
                'img2': img2_batch,
                'labels': labels_batch,
                'text': text_batch
            }
        else:
            # No text data
            img1_tensors, img2_tensors, labels = [], [], []
            
            for img1, img2, label in batch:
                img1_tensors.append(img1)
                img2_tensors.append(img2)
                labels.append(label)
            
            # Stack tensors
            img1_batch = torch.stack(img1_tensors)
            img2_batch = torch.stack(img2_tensors)
            labels_batch = torch.tensor(labels)
            
            return {
                'img1': img1_batch,
                'img2': img2_batch,
                'labels': labels_batch
            }