# src/rlhf_ui/ui/visualizers/metrics.py
"""
UI component for visualizing training metrics and model performance.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QComboBox, QTabWidget, QSizePolicy
)
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from rlhf_ui.models.trainer import RewardModelTrainer

logger = logging.getLogger(__name__)

class MatplotlibCanvas(FigureCanvas):
    """Matplotlib canvas for embedding in Qt."""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """
        Initialize the canvas.
        
        Args:
            parent: Parent widget
            width: Figure width in inches
            height: Figure height in inches
            dpi: Dots per inch
        """
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Setup
        self.fig.tight_layout()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()


class TrainingMetricsWidget(QWidget):
    """
    Widget for visualizing training metrics and testing reward model predictions.
    """
    
    def __init__(
        self,
        model_output_dir: Path,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.model_output_dir = Path(model_output_dir)
        
        # Setup UI first
        self._init_ui()
        
        # Initialize trainer
        self.trainer: Optional[RewardModelTrainer] = None
        self._init_trainer()
        
        # Load metrics data
        self.checkpoint_metrics: list[dict[str, float | str | Path]] = []
        self._load_checkpoint_metrics()
        
        # Update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._load_checkpoint_metrics)
        self.update_timer.start(10000)  # Update every 10 seconds
    
    def _init_trainer(self) -> None:
        """Initialize the reward model trainer."""
        try:
            # Dummy initialization - we'll load checkpoints later
            self.trainer = RewardModelTrainer(
                preference_file="dummy.csv",  # This won't be used
                image_folder=self.model_output_dir.parent / "images"
            )
        except Exception:
            logger.exception("Failed to initialize trainer")
            self.trainer = None
    
    def _load_checkpoint_metrics(self) -> None:
        """Load metrics from checkpoint metadata files."""
        if not self.model_output_dir.exists():
            return
            
        # Find all metadata files
        metadata_files = list(self.model_output_dir.glob("reward_model_epoch_*_metadata.json"))
        if not metadata_files:
            return
            
        # Load metrics from each file
        new_metrics = []
        for meta_file in metadata_files:
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                    
                # Get checkpoint path
                checkpoint_path = meta_file.with_suffix('').with_suffix('.safetensors')
                
                # Add to metrics list
                new_metrics.append({
                    'checkpoint': checkpoint_path,
                    'epoch': metadata.get('epoch', 0),
                    'loss': metadata.get('loss', 0.0),
                    'accuracy': metadata.get('accuracy', 0.0),
                    'timestamp': metadata.get('timestamp', ''),
                })
            except Exception as e:
                logger.error(f"Error loading metadata from {meta_file}: {e}")
        
        # Sort by epoch
        new_metrics.sort(key=lambda x: x['epoch'])
        
        # Update only if metrics have changed
        if new_metrics != self.checkpoint_metrics:
            self.checkpoint_metrics = new_metrics
            self._update_metrics_display()
    
    def _init_ui(self) -> None:
        """Initialize the UI components."""
        main_layout = QVBoxLayout(self)
        
        # Tabbed interface
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Create tabs
        self._create_metrics_tab()
        self._create_prediction_tab()
    
    def _create_metrics_tab(self) -> None:
        """Create the training metrics visualization tab."""
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)
        
        # Metrics selection
        controls_layout = QHBoxLayout()
        
        metric_label = QLabel("Metric:")
        controls_layout.addWidget(metric_label)
        
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["Loss", "Accuracy"])
        controls_layout.addWidget(self.metric_combo)
        
        controls_layout.addStretch()
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._load_checkpoint_metrics)
        controls_layout.addWidget(refresh_btn)
        
        metrics_layout.addLayout(controls_layout)
        
        # Add matplotlib canvas for plotting
        self.metrics_canvas = MatplotlibCanvas(self)
        metrics_layout.addWidget(self.metrics_canvas)
        
        # Connect signal after canvas is created
        self.metric_combo.currentIndexChanged.connect(self._update_metrics_display)
        
        metrics_layout.addWidget(self.metrics_canvas)
        self.tabs.addTab(metrics_widget, "Metrics")
        
    def _create_prediction_tab(self) -> None:
        """Create the prediction testing tab."""
        prediction_widget = QWidget()
        prediction_layout = QVBoxLayout(prediction_widget)
        # Add a placeholder label
        prediction_layout.addWidget(QLabel("Prediction testing interface coming soon..."))
        self.tabs.addTab(prediction_widget, "Predictions")
    
    def _update_metrics_display(self) -> None:
        """Update the metrics plot based on the selected metric."""
        if not self.checkpoint_metrics:
            return
            
        selected_metric = self.metric_combo.currentText().lower()
        epochs = [m['epoch'] for m in self.checkpoint_metrics]
        values = [m[selected_metric] for m in self.checkpoint_metrics]
        
        # Update plot
        self.metrics_canvas.axes.clear()
        self.metrics_canvas.axes.plot(epochs, values)
        self.metrics_canvas.axes.set_xlabel('Epoch')
        self.metrics_canvas.axes.set_ylabel(selected_metric.capitalize())
        self.metrics_canvas.draw()

