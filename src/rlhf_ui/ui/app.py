# src/rlhf_ui/ui/app.py
"""
Main application window for the RLHF UI application.
"""

import logging
from pathlib import Path

from PyQt6.QtCore import QSize, QSettings
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QToolBar, QStatusBar, QFileDialog,
    QMessageBox
)

from rlhf_ui.config import AppConfig
from rlhf_ui.ui.collectors.preference import PreferenceCollectorWidget
from rlhf_ui.ui.trainers.reward import RewardModelTrainerWidget
from rlhf_ui.ui.visualizers.metrics import TrainingMetricsWidget

logger = logging.getLogger(__name__)

class RLHFApplication(QMainWindow):
    """
    Main application window for the RLHF UI.
    """
    
    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        
        # Setup window
        self.setWindowTitle("RLHF UI - Human Preference Collection & Reward Model Training")
        self.resize(config.window_size[0], config.window_size[1])
        
        # Setup UI
        self._init_ui()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()
        
        # Load saved settings
        self._load_settings()
        
        logger.info("Application UI initialized")
    
    def _init_ui(self) -> None:
        """Initialize the main UI components."""
        # Central widget with tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # Create tabs
        self._create_preference_tab()
        self._create_training_tab()
        self._create_metrics_tab()
        
        # Set initial tab
        self.tabs.setCurrentIndex(0)
    
    def _create_preference_tab(self) -> None:
        """Create the preference collection tab."""
        self.preference_widget = PreferenceCollectorWidget(
            image_folder=self.config.image_folder,
            output_folder=self.config.output_folder,
            sampling_strategy=self.config.sampling_strategy,
            parent=self
        )
        self.tabs.addTab(self.preference_widget, "Collect Preferences")
    
    def _create_training_tab(self) -> None:
        """Create the model training tab."""
        self.training_widget = RewardModelTrainerWidget(
            preference_file=self.config.output_folder / "preferences.csv",
            image_folder=self.config.image_folder,
            model_output_dir=self.config.model_output_dir,
            parent=self
        )
        self.tabs.addTab(self.training_widget, "Train Reward Model")
    
    def _create_metrics_tab(self) -> None:
        """Create the training metrics visualization tab."""
        self.metrics_widget = TrainingMetricsWidget(
            model_output_dir=self.config.model_output_dir,
            parent=self
        )
        self.tabs.addTab(self.metrics_widget, "Training Metrics")
    
    def _setup_menu(self) -> None:
        """Setup the application menu bar."""
        # File menu
        file_menu = self.menuBar().addMenu("&File") # type: ignore
        
        # Open image folder
        open_images_action = QAction("Open Image &Folder...", self)
        open_images_action.setShortcut(QKeySequence.StandardKey.Open)
        open_images_action.triggered.connect(self._select_image_folder)
        file_menu.addAction(open_images_action) # type: ignore
        
        # Save preferences
        save_prefs_action = QAction("&Save Preferences", self)
        save_prefs_action.setShortcut(QKeySequence("Ctrl+S"))
        save_prefs_action.triggered.connect(self._save_preferences)
        file_menu.addAction(save_prefs_action) # type: ignore
        
        file_menu.addSeparator() # type: ignore
        
        # Export model
        export_model_action = QAction("&Export Model...", self)
        export_model_action.triggered.connect(self._export_model)
        file_menu.addAction(export_model_action) # type: ignore
        
        file_menu.addSeparator() # type: ignore
        
        # Exit
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action) # type: ignore
        
        # Edit menu
        edit_menu = self.menuBar().addMenu("&Edit") # type: ignore
        
        # Settings
        settings_action = QAction("&Settings...", self)
        settings_action.triggered.connect(self._show_settings)
        edit_menu.addAction(settings_action) # type: ignore
        
        # View menu
        view_menu = self.menuBar().addMenu("&View") # type: ignore
        
        # Switch tabs
        for i in range(self.tabs.count()):
            tab_action = QAction(f"Show {self.tabs.tabText(i)}", self)
            tab_action.setShortcut(QKeySequence(f"Ctrl+{i+1}"))
            tab_index = i  # Create a copy of i for the lambda
            tab_action.triggered.connect(lambda checked, idx=tab_index: self.tabs.setCurrentIndex(idx))
            view_menu.addAction(tab_action) # type: ignore
        
        # Help menu
        help_menu = self.menuBar().addMenu("&Help") # type: ignore
        
        # About
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action) # type: ignore
    
    def _setup_toolbar(self) -> None:
        """Setup the application toolbar."""
        self.toolbar = QToolBar("Main Toolbar")
        self.toolbar.setMovable(False)
        self.toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(self.toolbar)
        
        # Add collection action
        collect_action = QAction("Collect Preferences", self)
        collect_action.triggered.connect(lambda: self.tabs.setCurrentIndex(0))
        self.toolbar.addAction(collect_action)
        
        # Add training action
        train_action = QAction("Train Model", self)
        train_action.triggered.connect(lambda: self.tabs.setCurrentIndex(1))
        self.toolbar.addAction(train_action)
        
        # Add metrics action
        metrics_action = QAction("View Metrics", self)
        metrics_action.triggered.connect(lambda: self.tabs.setCurrentIndex(2))
        self.toolbar.addAction(metrics_action)
        
        self.toolbar.addSeparator()
        
        # Add export action
        export_action = QAction("Export Model", self)
        export_action.triggered.connect(self._export_model)
        self.toolbar.addAction(export_action)
    
    def _setup_statusbar(self) -> None:
        """Setup the application status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Ready")
    
    def _load_settings(self) -> None:
        """Load application settings."""
        settings = QSettings("Your Organization", "RLHF UI")
        if settings.contains("geometry"):
            self.restoreGeometry(settings.value("geometry"))
        if settings.contains("windowState"):
            self.restoreState(settings.value("windowState"))
        
        # Load last used directories
        if settings.contains("lastImageFolder"):
            last_folder = settings.value("lastImageFolder")
            if Path(last_folder).exists():
                self.config.image_folder = Path(last_folder)
    
    def _save_settings(self) -> None:
        """Save application settings."""
        settings = QSettings("Your Organization", "RLHF UI")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        settings.setValue("lastImageFolder", str(self.config.image_folder))
    
    def _select_image_folder(self) -> None:
        """Open dialog to select image folder."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Image Folder",
            str(self.config.image_folder),
            QFileDialog.Option.ShowDirsOnly
        )
        
        if folder:
            self.config.image_folder = Path(folder)
            self.statusbar.showMessage(f"Image folder set to: {folder}", 5000)
            
            # Confirm reload
            reply = QMessageBox.question(
                self,
                "Reload Images",
                "Do you want to reload the application with the new image folder?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self._reload_application()
    
    def _save_preferences(self) -> None:
        """Save current preferences data."""
        # This is handled automatically in the preference collector,
        # but we provide an explicit save option for peace of mind
        self.statusbar.showMessage("Preferences saved", 3000)
    
    def _export_model(self) -> None:
        """Export the trained reward model."""
        # Switch to training tab if not already there
        self.tabs.setCurrentIndex(1)
        
        # Delegate to training widget
        self.training_widget.export_model()
    
    def _show_settings(self) -> None:
        """Show settings dialog."""
        # TODO: Implement settings dialog
        self.statusbar.showMessage("Settings dialog not implemented yet", 3000)
    
    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About RLHF UI",
            "<h1>RLHF UI</h1>"
            "<p>Version 1.0</p>"
            "<p>A tool for collecting human preferences and training reward models for RLHF.</p>"
            "<p>&copy; 2025 Your Organization</p>"
        )
    
    def _reload_application(self) -> None:
        """Reload the application with new settings."""
        # Save settings before reload
        self._save_settings()
        
        # Remove existing tabs
        while self.tabs.count() > 0:
            self.tabs.removeTab(0)
        
        # Recreate tabs
        self._create_preference_tab()
        self._create_training_tab()
        self._create_metrics_tab()
        
        self.statusbar.showMessage("Application reloaded", 3000)
    
    def show_training_tab(self) -> None:
        """Switch to the training tab."""
        self.tabs.setCurrentIndex(1)
    
    def closeEvent(self, event) -> None:
        """Handle window close event."""
        # Save settings
        self._save_settings()
        
        # Check if we need to save anything
        if self.training_widget.is_training():
            reply = QMessageBox.question(
                self,
                "Exit Confirmation",
                "Training is in progress. Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
        
        # Accept close event
        event.accept()