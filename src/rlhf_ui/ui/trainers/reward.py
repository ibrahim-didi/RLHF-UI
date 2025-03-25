# src/rlhf_ui/ui/trainers/reward.py
"""
UI for training reward models based on collected preferences.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QProgressBar, QSpinBox, QDoubleSpinBox,
    QFormLayout, QGroupBox, QFileDialog, QCheckBox,
    QTextEdit, QSplitter, QMessageBox
)

from rlhf_ui.models.trainer import RewardModelTrainer

logger = logging.getLogger(__name__)

class TrainingThread(QThread):
    """Thread for running model training in the background."""
    
    # Signals
    progress_updated = pyqtSignal(int, str)
    epoch_completed = pyqtSignal(int, float, float)
    training_completed = pyqtSignal(str)
    training_error = pyqtSignal(str)
    
    def __init__(
        self,
        trainer: RewardModelTrainer,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        use_text_embeddings: bool
    ):
        super().__init__()
        self.trainer = trainer
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.use_text_embeddings = use_text_embeddings
    
    def run(self):
        """Run training in a separate thread."""
        try:
            # Set up a callback to report progress
            def progress_callback(epoch, progress, loss, accuracy):
                self.progress_updated.emit(progress, f"Epoch {epoch+1}/{self.epochs}")
                if progress == 100:  # Epoch complete
                    self.epoch_completed.emit(epoch, loss, accuracy)
            
            # Start training
            result = self.trainer.train(
                epochs=self.epochs,
                lr=self.learning_rate,
                batch_size=self.batch_size,
                text_embedding_size=768 if self.use_text_embeddings else 0,
                progress_callback=progress_callback
            )
            
            # Signal completion
            self.training_completed.emit(result)
            
        except Exception as e:
            logger.exception("Error during training")
            self.training_error.emit(str(e))


class RewardModelTrainerWidget(QWidget):
    """
    Widget for training reward models based on collected preferences.
    """
    
    def __init__(
        self,
        preference_file: Path,
        image_folder: Path,
        model_output_dir: Path = Path("reward_model"),
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.preference_file = Path(preference_file)
        self.image_folder = Path(image_folder)
        self.model_output_dir = Path(model_output_dir)
        self.model_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize trainer
        self.trainer: Optional[RewardModelTrainer] = None
        self._init_trainer()
        
        # State tracking
        self.training_thread: Optional[TrainingThread] = None
        self.is_training_active = False
        self.current_epoch = 0
        self.training_log: list[str] = []
        
        # Setup UI
        self._init_ui()
        self._update_ui_state()
        
        # Auto-update log timer
        self.log_timer = QTimer(self)
        self.log_timer.timeout.connect(self._update_log_display)
        self.log_timer.start(1000)  # Update every second
    
    def _init_trainer(self) -> None:
        """Initialize the reward model trainer."""
        if self.preference_file.exists():
            try:
                self.trainer = RewardModelTrainer(
                    str(self.preference_file),
                    str(self.image_folder),
                    str(self.model_output_dir)
                )
                logger.info(f"Initialized trainer with {len(self.trainer.preferences_df)} preference records")
            except Exception:
                logger.exception("Failed to initialize trainer")
                self.trainer = None
    
    def _init_ui(self) -> None:
        """Initialize the UI components."""
        main_layout = QVBoxLayout(self)
        
        # Split into config and results sections
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # ---- Configuration section ----
        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)
        config_layout.setContentsMargins(0, 0, 0, 0)
        
        # Data info
        info_group = QGroupBox("Data Information")
        info_layout = QFormLayout(info_group)
        
        self.data_count_label = QLabel("No data available")
        info_layout.addRow("Preference records:", self.data_count_label)
        
        self.images_count_label = QLabel("Unknown")
        info_layout.addRow("Images available:", self.images_count_label)
        
        self.model_status_label = QLabel("Not trained")
        info_layout.addRow("Model status:", self.model_status_label)
        
        config_layout.addWidget(info_group)
        
        # Training parameters
        params_group = QGroupBox("Training Parameters")
        params_layout = QFormLayout(params_group)
        
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 100)
        self.epochs_spinbox.setValue(20)
        params_layout.addRow("Epochs:", self.epochs_spinbox)
        
        self.lr_spinbox = QDoubleSpinBox()
        self.lr_spinbox.setRange(0.00001, 0.1)
        self.lr_spinbox.setSingleStep(0.0001)
        self.lr_spinbox.setDecimals(5)
        self.lr_spinbox.setValue(0.0001)
        params_layout.addRow("Learning rate:", self.lr_spinbox)
        
        self.batch_size_spinbox = QSpinBox()
        self.batch_size_spinbox.setRange(1, 64)
        self.batch_size_spinbox.setValue(8)
        params_layout.addRow("Batch size:", self.batch_size_spinbox)
        
        self.use_text_checkbox = QCheckBox("Use text prompts if available")
        self.use_text_checkbox.setChecked(False)
        params_layout.addRow("", self.use_text_checkbox)
        
        self.use_gpu_checkbox = QCheckBox("Use GPU if available")
        self.use_gpu_checkbox.setChecked(True)
        params_layout.addRow("", self.use_gpu_checkbox)
        
        config_layout.addWidget(params_group)
        
        # Training controls
        controls_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton("Refresh Data")
        self.refresh_btn.clicked.connect(self._refresh_data)
        controls_layout.addWidget(self.refresh_btn)
        
        controls_layout.addStretch()
        
        self.train_btn = QPushButton("Start Training")
        self.train_btn.clicked.connect(self._start_training)
        self.train_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        controls_layout.addWidget(self.train_btn)
        
        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.clicked.connect(self._stop_training)
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        self.stop_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_btn)
        
        self.export_btn = QPushButton("Export Model")
        self.export_btn.clicked.connect(self.export_model)
        controls_layout.addWidget(self.export_btn)
        
        config_layout.addLayout(controls_layout)
        
        # ---- Results section ----
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(0, 0, 0, 0)
        
        # Progress display
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("Not training")
        progress_layout.addWidget(self.progress_label)
        
        results_layout.addWidget(progress_group)
        
        # Training log
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        results_layout.addWidget(log_group)
        
        # Add widgets to splitter
        splitter.addWidget(config_widget)
        splitter.addWidget(results_widget)
        splitter.setSizes([300, 500])
        
        main_layout.addWidget(splitter)
        
        # Initialize data display
        self._refresh_data()
    
    def _refresh_data(self) -> None:
        """Refresh data information display."""
        if self.trainer is None:
            self._init_trainer()
        
        if self.trainer is not None:
            # Update preference count
            record_count = len(self.trainer.preferences_df)
            self.data_count_label.setText(f"{record_count} records")
            
            # Update images count
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
            image_count = sum(1 for f in self.image_folder.glob('**/*') if f.suffix.lower() in image_extensions)
            self.images_count_label.setText(f"{image_count} images")
            
            # Check model status
            model_files = list(self.model_output_dir.glob("reward_model_epoch_*.safetensors"))
            if model_files:
                self.model_status_label.setText(f"Trained ({len(model_files)} checkpoints)")
                self.export_btn.setEnabled(True)
            else:
                self.model_status_label.setText("Not trained")
                self.export_btn.setEnabled(False)
        else:
            self.data_count_label.setText("No data available")
            self.images_count_label.setText("Unknown")
            self.model_status_label.setText("Not trained")
            self.export_btn.setEnabled(False)
    
    def _update_ui_state(self) -> None:
        """Update UI element states based on the current application state."""
        is_training = self.is_training_active
        
        # Update button states
        self.train_btn.setEnabled(not is_training and self.trainer is not None)
        self.stop_btn.setEnabled(is_training)
        self.refresh_btn.setEnabled(not is_training)
        
        # Update parameter widgets
        self.epochs_spinbox.setEnabled(not is_training)
        self.lr_spinbox.setEnabled(not is_training)
        self.batch_size_spinbox.setEnabled(not is_training)
        self.use_text_checkbox.setEnabled(not is_training)
        self.use_gpu_checkbox.setEnabled(not is_training)
    
    def _start_training(self) -> None:
        """Start the training process."""
        if self.trainer is None or self.is_training_active:
            return
        
        # Confirm if training data seems insufficient
        # TODO: maybe a better approach is to check in context of number of pair combinations
        record_count = len(self.trainer.preferences_df)
        if record_count < 50:
            reply = QMessageBox.question(
                self,
                "Limited Training Data",
                f"You only have {record_count} preference records. This may not be enough for good results. "
                "Do you want to continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.No:
                return
        
        # Get parameters
        epochs = self.epochs_spinbox.value()
        learning_rate = self.lr_spinbox.value()
        batch_size = self.batch_size_spinbox.value()
        use_text = self.use_text_checkbox.isChecked()
        use_gpu = self.use_gpu_checkbox.isChecked()
        
        # Setup device
        if use_gpu and torch.cuda.is_available():
            torch.cuda.set_device(0)
        else:
            # Force CPU
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        
        # Clear log and reset progress
        self.training_log = []
        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"Preparing to train for {epochs} epochs...")
        
        # Log start
        start_msg = f"Starting training with {record_count} preference records\n"
        start_msg += f"Parameters: epochs={epochs}, lr={learning_rate}, batch_size={batch_size}, use_text={use_text}\n"
        start_msg += f"Using {'GPU' if use_gpu and torch.cuda.is_available() else 'CPU'} for training\n"
        self._add_to_log(start_msg)
        
        # Create and start training thread
        self.training_thread = TrainingThread(
            self.trainer,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            use_text_embeddings=use_text
        )
        
        # Connect signals
        self.training_thread.progress_updated.connect(self._on_progress_update)
        self.training_thread.epoch_completed.connect(self._on_epoch_completed)
        self.training_thread.training_completed.connect(self._on_training_completed)
        self.training_thread.training_error.connect(self._on_training_error)
        
        # Start training
        self.training_thread.start()
        self.is_training_active = True
        self.current_epoch = 0
        
        # Update UI
        self._update_ui_state()
        
        # Log
        logger.info(f"Started training with {epochs} epochs")
    
    def _stop_training(self) -> None:
        """Stop the training process."""
        if not self.is_training_active or self.training_thread is None:
            return
        
        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Stop Training",
            "Are you sure you want to stop the training process? "
            "Progress in the current epoch will be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Terminate thread
            self.training_thread.terminate()
            self.training_thread.wait()
            
            # Update state
            self.is_training_active = False
            self._update_ui_state()
            
            # Update UI
            self.progress_label.setText("Training stopped by user")
            self._add_to_log("Training was stopped by user")
            logger.info("Training stopped by user")
    
    def _on_progress_update(self, progress: int, status: str) -> None:
        """
        Handle progress updates from the training thread.
        
        Args:
            progress: Progress percentage (0-100)
            status: Status message
        """
        self.progress_bar.setValue(progress)
        self.progress_label.setText(status)
    
    def _on_epoch_completed(self, epoch: int, loss: float, accuracy: float) -> None:
        """
        Handle epoch completion from the training thread.
        
        Args:
            epoch: Completed epoch number (0-based)
            loss: Training loss
            accuracy: Training accuracy
        """
        epoch_num = epoch + 1
        self.current_epoch = epoch_num
        
        # Log epoch results
        log_msg = f"Epoch {epoch_num} completed: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}"
        self._add_to_log(log_msg)
        
        # Update model status
        self.model_status_label.setText(f"Training (epoch {epoch_num})")
    
    def _on_training_completed(self, result: str) -> None:
        """
        Handle training completion from the training thread.
        
        Args:
            result: Result message or path to trained model
        """
        self.is_training_active = False
        self._update_ui_state()
        
        # Update UI
        self.progress_label.setText("Training completed successfully!")
        self.progress_bar.setValue(100)
        
        # Log completion
        self._add_to_log("\nTraining completed successfully!")
        self._add_to_log(f"Model saved to: {result}")
        
        # Update model status
        self._refresh_data()
        
        # Show completion dialog
        QMessageBox.information(
            self,
            "Training Completed",
            "The reward model training has completed successfully!\n\n"
            "You can now export the model for use in RLHF fine-tuning.",
            QMessageBox.StandardButton.Ok
        )
        
        # Switch to metrics tab
        parent = self.parent()
        if parent is not None and hasattr(parent, 'tabs'):
            parent.tabs.setCurrentIndex(2)
    
    def _on_training_error(self, error_msg: str) -> None:
        """
        Handle training errors from the training thread.
        
        Args:
            error_msg: Error message
        """
        self.is_training_active = False
        self._update_ui_state()
        
        # Update UI
        self.progress_label.setText("Training failed")
        
        # Log error
        self._add_to_log("\nERROR: Training failed with the following error:")
        self._add_to_log(error_msg)
        
        # Show error dialog
        QMessageBox.critical(
            self,
            "Training Error",
            f"An error occurred during training:\n\n{error_msg}",
            QMessageBox.StandardButton.Ok
        )
    
    def _add_to_log(self, message: str) -> None:
        """Add a message to the training log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.training_log.append(log_entry)
        
        # Don't update UI directly if called from training thread
        if QThread.currentThread() is self.thread():
            self._update_log_display()
    
    def _update_log_display(self) -> None:
        """Update the log text display."""
        if not hasattr(self, 'log_text') or self.log_text is None:
            return
            
        # Set log contents
        self.log_text.setText("\n".join(self.training_log))
        
        # Scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())
    
    def export_model(self) -> None:
        """Export the trained model for RLHF fine-tuning."""
        if self.trainer is None:
            QMessageBox.warning(
                self,
                "Export Error",
                "No trainer initialized. Cannot export model.",
                QMessageBox.StandardButton.Ok
            )
            return
        
        # Check if model exists
        model_files = list(self.model_output_dir.glob("reward_model_epoch_*.safetensors"))
        if not model_files:
            QMessageBox.warning(
                self,
                "Export Error",
                "No trained models found. Please train the model first.",
                QMessageBox.StandardButton.Ok
            )
            return
        
        # Get export path from user
        export_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Reward Model",
            str(self.model_output_dir / "reward_model_for_rlhf.safetensors"),
            "SafeTensors files (*.safetensors)"
        )
        
        if not export_path:
            return
        
        try:
            # Export model
            result_path = self.trainer.export_for_rlhf(Path(export_path))
            
            # Show success message
            QMessageBox.information(
                self,
                "Export Successful",
                f"Reward model exported to:\n{result_path}\n\n"
                "The model is now ready for use in RLHF fine-tuning.",
                QMessageBox.StandardButton.Ok
            )
            
            # Add to log
            self._add_to_log(f"Model exported to: {result_path}")
            
        except Exception as e:
            logger.exception("Error exporting model")
            QMessageBox.critical(
                self,
                "Export Error",
                f"An error occurred during model export:\n\n{str(e)}",
                QMessageBox.StandardButton.Ok
            )
    
    def is_training(self) -> bool:
        """Check if training is currently active."""
        return self.is_training_active