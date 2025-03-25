# src/rlhf_ui/ui/collectors/preference.py
"""
Preference collection UI component using PyQt6.
"""

import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QProgressBar, QMessageBox
)

from rlhf_ui.data.sampler import ImagePairSampler
from rlhf_ui.ui.components.image_pair import ImagePairWidget

logger = logging.getLogger(__name__)

class PreferenceCollectorWidget(QWidget):
    """
    Widget for collecting human preferences between pairs of images.
    """
    
    def __init__(
        self,
        image_folder: Path,
        output_folder: Path = Path("preference_data"),
        sampling_strategy: str = "active",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.image_folder = Path(image_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True, parents=True)
        
        # Setup preference database
        self.preference_file = self.output_folder / "preferences.csv"
        self._initialize_preference_database()
        
        # Setup sampler
        self.sampler = ImagePairSampler(
            self.image_folder,
            self.preferences_df,
            strategy=sampling_strategy
        )
        
        # UI state
        self.current_pair: Optional[tuple[Path, Path]] = None
        self.prompt = ""
        self.comparison_start_time = time.time()
        self.total_comparisons = len(self.preferences_df)
        self.session_completed = False
        
        # Setup UI
        self._init_ui()
        
        # Load initial pair
        self._load_next_pair()
    
    def _initialize_preference_database(self) -> None:
        """Create or load the preference database."""
        if not self.preference_file.exists():
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
        else:
            self.preferences_df = pd.read_csv(self.preference_file)
            
        logger.info(f"Loaded preference database with {len(self.preferences_df)} records")
    
    def _init_ui(self) -> None:
        """Initialize the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        
        # Instructions
        instructions = QLabel(
            "Which image better matches the prompt? If no prompt, which is better quality?"
        )
        instructions.setStyleSheet("font-size: 16px; font-weight: bold;")
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(instructions)
        
        # Prompt display
        self.prompt_label = QLabel("(No specific prompt)")
        self.prompt_label.setStyleSheet("font-size: 14px;")
        self.prompt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.prompt_label)
        
        # Image pair display
        self.image_pair_widget = ImagePairWidget()
        main_layout.addWidget(self.image_pair_widget, 1)  # Stretch to fill available space
        
        # Preference buttons
        buttons_layout = QHBoxLayout()
        
        self.prefer_left_btn = QPushButton("Prefer Left (A)")
        self.prefer_left_btn.setStyleSheet("font-size: 14px; padding: 10px;")
        self.prefer_left_btn.clicked.connect(lambda: self._on_preference(1))
        buttons_layout.addWidget(self.prefer_left_btn)
        
        self.prefer_right_btn = QPushButton("Prefer Right (B)")
        self.prefer_right_btn.setStyleSheet("font-size: 14px; padding: 10px;")
        self.prefer_right_btn.clicked.connect(lambda: self._on_preference(2))
        buttons_layout.addWidget(self.prefer_right_btn)
        
        main_layout.addLayout(buttons_layout)
        
        # Additional controls
        controls_layout = QHBoxLayout()
        
        self.tie_btn = QPushButton("Equal / Can't Decide (E)")
        self.tie_btn.clicked.connect(lambda: self._on_preference(0))
        controls_layout.addWidget(self.tie_btn)
        
        self.skip_btn = QPushButton("Skip Pair (S)")
        self.skip_btn.clicked.connect(self._load_next_pair)
        controls_layout.addWidget(self.skip_btn)
        
        controls_layout.addStretch()
        
        self.finish_btn = QPushButton("Finish Rating (F)")
        self.finish_btn.clicked.connect(self._finish_rating)
        self.finish_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        controls_layout.addWidget(self.finish_btn)
        
        main_layout.addLayout(controls_layout)
        
        # Progress and stats bar
        stats_bar = QHBoxLayout()
        
        self.stats_label = QLabel(f"Total comparisons: {self.total_comparisons}")
        stats_bar.addWidget(self.stats_label)
        
        stats_bar.addStretch()
        
        self.progress_label = QLabel("Progress:")
        stats_bar.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)  # Target 100 comparisons by default
        self.progress_bar.setValue(self.total_comparisons)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / 100")
        stats_bar.addWidget(self.progress_bar)
        
        main_layout.addLayout(stats_bar)
        
        # Setup keyboard shortcuts
        QShortcut(QKeySequence("A"), self).activated.connect(lambda: self._on_preference(1))
        QShortcut(QKeySequence("B"), self).activated.connect(lambda: self._on_preference(2))
        QShortcut(QKeySequence("E"), self).activated.connect(lambda: self._on_preference(0))
        QShortcut(QKeySequence("S"), self).activated.connect(self._load_next_pair)
        QShortcut(QKeySequence("F"), self).activated.connect(self._finish_rating)
    
    def _load_next_pair(self) -> None:
        """Load the next pair of images for comparison."""
        if self.session_completed:
            return
            
        # Sample a new pair
        img1, img2 = self.sampler.sample_pair()
        self.current_pair = (img1, img2)
        
        # Optional: set a prompt (could be based on filenames or metadata)
        self.prompt = ""
        self.prompt_label.setText(self.prompt if self.prompt else "(No specific prompt)")
        
        # Load and display images
        self.image_pair_widget.set_images(img1, img2)
        
        # Reset timer
        self.comparison_start_time = time.time()
        
        # Update stats
        self.stats_label.setText(f"Total comparisons: {self.total_comparisons}")
        self.progress_bar.setValue(min(self.total_comparisons, 100))
    
    def _on_preference(self, preference: int) -> None:
        """
        Handle preference selection (1 for left, 2 for right, 0 for tie).
        
        Args:
            preference: Preference value (1=left, 2=right, 0=tie/skip)
        """
        if self.session_completed or self.current_pair is None:
            return
            
        response_time_ms = (time.time() - self.comparison_start_time) * 1000
        self._record_preference(
            self.current_pair[0],
            self.current_pair[1],
            preference,
            response_time_ms
        )
        self._load_next_pair()
    
    def _record_preference(
        self, 
        image1: Path, 
        image2: Path, 
        preferred: int, 
        response_time_ms: float,
        rater_id: str = "default"
    ) -> None:
        """
        Record a human preference between two images.
        
        Args:
            image1: Path to first image
            image2: Path to second image
            preferred: 1 if image1 is preferred, 2 if image2, 0 if tie/skip
            response_time_ms: Response time in milliseconds
            rater_id: ID of the human rater
        """
        new_row = pd.DataFrame({
            'image1': [str(image1)],
            'image2': [str(image2)],
            'preferred': [preferred],
            'prompt': [self.prompt],
            'timestamp': [datetime.now().isoformat()],
            'rater_id': [rater_id],
            'response_time_ms': [response_time_ms]
        })
        
        self.preferences_df = pd.concat([self.preferences_df, new_row], ignore_index=True)
        self.preferences_df.to_csv(self.preference_file, index=False)
        self.total_comparisons += 1
    
    def _finish_rating(self) -> None:
        """Finish the rating session and emit signal."""
        if self.session_completed:
            return
            
        logger.info(f"Rating session completed with {self.total_comparisons} comparisons")
        self.session_completed = True
        
        # Show completion dialog
        QMessageBox.information(
            self,
            "Session Complete",
            f"Rating session completed with {self.total_comparisons} comparisons.\n\n"
            "Would you like to proceed to training the reward model?",
            QMessageBox.StandardButton.Ok
        )
        
        parent = self.parent()
        if parent and hasattr(parent, 'show_training_tab'):
            parent.show_training_tab()  # Switch to training tab