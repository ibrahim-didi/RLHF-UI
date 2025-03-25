# src/rlhf_ui/ui/components/buttons.py
"""
Reusable button components for the RLHF UI application.
"""

import logging
from typing import Optional, Callable

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QPushButton, QWidget, QToolButton, QSizePolicy,
    QHBoxLayout, QVBoxLayout, QLabel, QBoxLayout
)

logger = logging.getLogger(__name__)

class ActionButton(QPushButton):
    """
    Button for primary actions with customizable styling.
    """
    
    def __init__(
        self,
        text: str,
        parent: Optional[QWidget] = None,
        icon: Optional[QIcon] = None,
        shortcut_key: Optional[str] = None,
        tooltip: Optional[str] = None,
        is_primary: bool = False,
        is_destructive: bool = False,
        is_disabled: bool = False,
        on_click: Optional[Callable] = None
    ):
        """
        Initialize the action button.
        
        Args:
            text: Button text
            parent: Parent widget
            icon: Button icon
            shortcut_key: Keyboard shortcut (e.g., "Ctrl+S" or "A")
            tooltip: Button tooltip
            is_primary: Whether this is a primary action button
            is_destructive: Whether this is a destructive action button
            is_disabled: Whether the button should be initially disabled
            on_click: Callback function when button is clicked
        """
        super().__init__(text, parent)
        
        # Set icon if provided
        if icon:
            self.setIcon(icon)
            self.setIconSize(QSize(16, 16))
        
        # Set tooltip if provided
        if tooltip:
            self.setToolTip(tooltip)
        
        # Set shortcut if provided
        if shortcut_key:
            self._shortcut = QShortcut(QKeySequence(shortcut_key), self.parent() or self)
            self._shortcut.activated.connect(self.click)
            
            # Add shortcut to tooltip
            current_tooltip = self.toolTip()
            if current_tooltip:
                self.setToolTip(f"{current_tooltip} ({shortcut_key})")
            else:
                self.setToolTip(f"Shortcut: {shortcut_key}")
        
        # Apply styling based on type
        if is_primary:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:pressed {
                    background-color: #398e3d;
                }
                QPushButton:disabled {
                    background-color: #cccccc;
                    color: #666666;
                }
            """)
        elif is_destructive:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    font-weight: bold;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #d32f2f;
                }
                QPushButton:pressed {
                    background-color: #b71c1c;
                }
                QPushButton:disabled {
                    background-color: #cccccc;
                    color: #666666;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background-color: #e0e0e0;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #d5d5d5;
                }
                QPushButton:pressed {
                    background-color: #c2c2c2;
                }
                QPushButton:disabled {
                    background-color: #f0f0f0;
                    color: #a0a0a0;
                }
            """)
        
        # Set initial state
        self.setEnabled(not is_disabled)
        
        # Connect click handler if provided
        if on_click:
            self.clicked.connect(on_click)


class PreferenceButton(QWidget):
    """
    Button for preference selection with label and shortcut.
    """
    
    def __init__(
        self,
        text: str,
        shortcut_key: str,
        parent: Optional[QWidget] = None,
        icon: Optional[QIcon] = None,
        is_primary: bool = False,
        on_click: Optional[Callable] = None
    ):
        """
        Initialize the preference button.
        
        Args:
            text: Button text
            shortcut_key: Keyboard shortcut (single key like "A")
            parent: Parent widget
            icon: Button icon
            is_primary: Whether this is a primary button
            on_click: Callback function when button is clicked
        """
        super().__init__(parent)
        
        # Create layout
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create button
        self.button = QPushButton(text)
        if icon:
            self.button.setIcon(icon)
            self.button.setIconSize(QSize(24, 24))
        
        # Apply styling
        if is_primary:
            self.button.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    border: none;
                    padding: 12px 24px;
                    border-radius: 4px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
                QPushButton:pressed {
                    background-color: #398e3d;
                }
                QPushButton:disabled {
                    background-color: #cccccc;
                    color: #666666;
                }
            """)
        else:
            self.button.setStyleSheet("""
                QPushButton {
                    background-color: #f0f0f0;
                    border: none;
                    padding: 12px 24px;
                    border-radius: 4px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #e5e5e5;
                }
                QPushButton:pressed {
                    background-color: #d0d0d0;
                }
                QPushButton:disabled {
                    background-color: #f0f0f0;
                    color: #a0a0a0;
                }
            """)
            
        # Create shortcut label
        shortcut_label = QLabel(f"Shortcut: {shortcut_key}")
        shortcut_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        shortcut_label.setStyleSheet("color: #666666; font-size: 11px;")
        
        # Set shortcut
        self._shortcut = QShortcut(QKeySequence(shortcut_key), self.parent() or self)
        if on_click:
            self._shortcut.activated.connect(on_click)
            self.button.clicked.connect(on_click)
        
        # Add to layout
        layout.addWidget(self.button)
        layout.addWidget(shortcut_label)
        
        # Make sure the button expands but not the label
        self.button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        shortcut_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)


class IconButton(QToolButton):
    """
    Small icon button for toolbar actions.
    """
    
    def __init__(
        self,
        icon: QIcon,
        tooltip: str,
        parent: Optional[QWidget] = None,
        shortcut_key: Optional[str] = None,
        on_click: Optional[Callable] = None
    ):
        """
        Initialize the icon button.
        
        Args:
            icon: Button icon
            tooltip: Button tooltip
            parent: Parent widget
            shortcut_key: Keyboard shortcut (e.g., "Ctrl+S")
            on_click: Callback function when button is clicked
        """
        super().__init__(parent)
        
        # Set icon
        self.setIcon(icon)
        self.setIconSize(QSize(24, 24))
        
        # Set tooltip
        if shortcut_key:
            self.setToolTip(f"{tooltip} ({shortcut_key})")
        else:
            self.setToolTip(tooltip)
        
        # Set size
        self.setFixedSize(QSize(32, 32))
        
        # Styling
        self.setStyleSheet("""
            QToolButton {
                border: none;
                background-color: transparent;
                border-radius: 4px;
            }
            QToolButton:hover {
                background-color: rgba(0, 0, 0, 0.1);
            }
            QToolButton:pressed {
                background-color: rgba(0, 0, 0, 0.2);
            }
        """)
        
        # Set shortcut if provided
        if shortcut_key:
            self._shortcut = QShortcut(QKeySequence(shortcut_key), self.parent() or self)
            if on_click:
                self._shortcut.activated.connect(on_click)
        
        # Connect click handler if provided
        if on_click:
            self.clicked.connect(on_click)


class ButtonGroup(QWidget):
    """
    Group of related buttons arranged horizontally or vertically.
    """
    
    def __init__(
        self,
        buttons: list,
        orientation: str = 'horizontal',
        spacing: int = 8,
        parent: Optional[QWidget] = None
    ):
        """
        Initialize the button group.
        
        Args:
            buttons: List of button widgets
            orientation: 'horizontal' or 'vertical'
            spacing: Spacing between buttons
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Create layout based on orientation
        layout_manager: QBoxLayout
        if orientation.lower() == 'vertical':
            layout_manager = QVBoxLayout(self)
        else:
            layout_manager = QHBoxLayout(self)
            
        # Set spacing
        layout_manager.setSpacing(spacing)
        layout_manager.setContentsMargins(0, 0, 0, 0)
        
        # Add buttons to layout
        for button in buttons:
            layout_manager.addWidget(button)