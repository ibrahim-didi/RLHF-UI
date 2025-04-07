"""
Preference Collection tab for the RLHF UI application.

This module contains the implementation of the Preference Collection tab, which allows
users to collect and record preferences between pairs of images.
"""

from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import logging

import gradio as gr
from PIL import Image

from rlhf_ui.config import AppConfig
from rlhf_ui.data.storage import PreferenceDataStorage
from rlhf_ui.data.sampler import ImagePairSampler
from rlhf_ui.ui.tabs.base import BaseTab

# Get logger for this module
logger = logging.getLogger(__name__)

class PreferenceCollectionTab(BaseTab):
    """
    Tab for collecting human preferences on pairs of images.
    
    This tab allows users to:
    - View pairs of images
    - Express preferences between them
    - Specify optional prompts or context for the pair
    - Track preference collection statistics
    """
    
    def __init__(self, config: AppConfig):
        """
        Initialize the preference collection tab.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.storage = PreferenceDataStorage(config.output_folder)
        self.sampler = ImagePairSampler(
            config.image_folder,
            self.storage.get_preferences(),
            strategy=config.sampling_strategy
        )
        
        # UI components (will be set during build)
        self.image_1 = None
        self.image_2 = None
        self.prompt_input = None
        self.prefer_1_btn = None
        self.prefer_2_btn = None
        self.tie_btn = None
        self.skip_btn = None
        self.load_pair_btn = None
        self.preference_count_text = None
        self.stats_text = None
        self.image_folder_input = None
        self.update_folder_btn = None
        self.sampling_strategy = None
        
        # Current image pair
        self.current_pair = None
    
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
                    ## Human Preference Collection
                    
                    Select the image you prefer based on quality, aesthetics, or the prompt.
                    Your preferences will be used to train a reward model.
                    """
                )
                
                # Prompt input
                self.prompt_input = gr.Textbox(
                    label="Prompt/Context (optional)",
                    placeholder="Enter any prompt or context for this image pair",
                    lines=2
                )
                
                # Image folder selection
                with gr.Row():
                    # Current folder display
                    folder_display = gr.Markdown(f"**Current Image Folder**: {str(self.config.image_folder)}")
                
                with gr.Row():
                    # Textbox for folder path
                    folder_path_input = gr.Textbox(
                        label="Image Folder Path",
                        placeholder="Enter the full path to your image folder",
                        value=str(self.config.image_folder)
                    )
                    
                with gr.Row():
                    # Instructions for folder selection
                    gr.Markdown("""
                    **Note**: Enter the full path to your image folder in the textbox above.
                    Example: `/home/user/images` or `C:\\Users\\user\\Pictures`
                    """)
                
                with gr.Row():
                    # Load button for the folder
                    load_dir_btn = gr.Button("Load Folder", variant="primary")
                
                # Sampling strategy selection
                self.sampling_strategy = gr.Dropdown(
                    label="Sampling Strategy",
                    choices=["random", "active", "diversity"],
                    value=self.config.sampling_strategy
                )
                
                # Progress display
                self.preference_count_text = gr.Markdown(
                    f"**Total preferences collected**: {self.storage.get_preference_count()}"
                )
                
                # Display stats
                stats_expander = gr.Accordion("Collection Statistics", open=False)
                with stats_expander:
                    self.stats_text = gr.Markdown(self.storage.get_data_summary())
        
        # Image pair display
        with gr.Row():
            with gr.Column():
                self.image_1 = gr.Image(
                    label="Image A",
                    type="pil",
                    elem_classes=["gallery-image"]
                )
                self.prefer_1_btn = gr.Button("Prefer Image A (1)", elem_classes=["feedback-button"])
            
            with gr.Column():
                self.image_2 = gr.Image(
                    label="Image B",
                    type="pil",
                    elem_classes=["gallery-image"]
                )
                self.prefer_2_btn = gr.Button("Prefer Image B (2)", elem_classes=["feedback-button"])
        
        with gr.Row():
            self.tie_btn = gr.Button("Equal / Can't Decide (0)")
            self.skip_btn = gr.Button("Skip Pair (S)")
        
        # Load initial pair button
        self.load_pair_btn = gr.Button("Load Next Pair", variant="primary")
        
        # Register event handlers for directory selection
        load_dir_btn.click(
            fn=self.update_folder,
            inputs=[folder_path_input],
            outputs=[folder_display, self.image_1, self.image_2]
        )
    
    def register_event_handlers(self) -> None:
        """
        Register event handlers for the tab's components.
        """
        self.sampling_strategy.change(
            self.update_sampling_strategy,
            inputs=[self.sampling_strategy],
            outputs=[gr.Markdown()]
        )
        
        self.load_pair_btn.click(
            self.load_image_pair,
            outputs=[self.image_1, self.image_2]
        )
        
        self.prefer_1_btn.click(
            lambda prompt: self.record_preference(1, prompt),
            inputs=[self.prompt_input],
            outputs=[gr.Markdown(), self.preference_count_text, self.stats_text, self.image_1, self.image_2]
        )
        
        self.prefer_2_btn.click(
            lambda prompt: self.record_preference(2, prompt),
            inputs=[self.prompt_input],
            outputs=[gr.Markdown(), self.preference_count_text, self.stats_text, self.image_1, self.image_2]
        )
        
        self.tie_btn.click(
            lambda prompt: self.record_preference(0, prompt),
            inputs=[self.prompt_input],
            outputs=[gr.Markdown(), self.preference_count_text, self.stats_text, self.image_1, self.image_2]
        )
        
        self.skip_btn.click(
            self.load_image_pair,
            outputs=[self.image_1, self.image_2]
        )
    
    def update_folder(self, folder_path) -> Tuple[str, Optional[Image.Image], Optional[Image.Image]]:
        """
        Update the image folder path.
        
        Args:
            folder_path: Path to the image folder
            
        Returns:
            Tuple with updated status message and new image pair
        """
        try:
            # Validate folder path
            path = Path(folder_path)
            if not path.exists() or not path.is_dir():
                return f"**Current Image Folder**: {str(self.config.image_folder)} (Invalid selection)", None, None
            
            # Update config
            self.config.image_folder = path
            
            # Reinitialize sampler
            self.sampler = ImagePairSampler(
                self.config.image_folder,
                self.storage.get_preferences(),
                strategy=self.config.sampling_strategy
            )
            
            # Load a new pair of images
            img1, img2 = self.load_image_pair()
            
            return f"**Current Image Folder**: {str(path)}", img1, img2
        except Exception as e:
            logger.error(f"Error updating folder: {e}")
            return f"**Current Image Folder**: {str(self.config.image_folder)} (Error: {str(e)})", None, None
    
    def update_sampling_strategy(self, strategy: str) -> str:
        """
        Update the sampling strategy.
        
        Args:
            strategy: New sampling strategy
            
        Returns:
            Status message
        """
        self.config.sampling_strategy = strategy
        self.sampler = ImagePairSampler(
            self.config.image_folder,
            self.storage.get_preferences(),
            strategy=strategy
        )
        return f"Updated sampling strategy to: {strategy}"
    
    def load_image_pair(self) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
        """
        Load a new image pair for preference collection.
        
        Returns:
            Tuple of two images
        """
        try:
            img1_path, img2_path = self.sampler.sample_pair()
            self.current_pair = (img1_path, img2_path)
            
            img1 = Image.open(img1_path)
            img2 = Image.open(img2_path)
            
            return img1, img2
        except Exception as e:
            logger.error(f"Error loading image pair: {e}")
            return None, None
    
    def record_preference(self, preference: int, prompt: str) -> Tuple[str, str, str, Optional[Image.Image], Optional[Image.Image]]:
        """
        Record a preference between two images.
        
        Args:
            preference: 1 for first image, 2 for second image, 0 for tie
            prompt: Optional prompt or context for the pair
            
        Returns:
            Tuple with status message, updated count text, stats, and new image pair
        """
        if self.current_pair is None:
            return "Please load a pair first", self.preference_count_text.value, "", None, None
        
        img1_path, img2_path = self.current_pair
        
        # Record preference
        self.storage.add_preference(
            image1=img1_path,
            image2=img2_path,
            preferred=preference,
            prompt=prompt
        )
        
        # Update counter
        count = self.storage.get_preference_count()
        count_text = f"**Total preferences collected**: {count}"
        
        # Update stats
        stats = self.storage.get_data_summary()
        
        # Load next pair
        img1, img2 = self.load_image_pair()
        
        return f"Preference recorded: {'A' if preference == 1 else 'B' if preference == 2 else 'Tie/Skip'}", count_text, stats, img1, img2 