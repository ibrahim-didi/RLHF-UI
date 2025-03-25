from PyQt6.QtWidgets import QWidget, QFrame, QLabel, QSizePolicy, QHBoxLayout, QVBoxLayout, QSplitter
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from pathlib import Path

class ImagePairWidget(QWidget):
    """Widget for displaying and comparing image pairs."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        
    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Left image
        self.left_frame = QFrame()
        self.left_frame.setFrameShape(QFrame.Shape.StyledPanel)
        left_layout = QVBoxLayout(self.left_frame)
        
        self.left_image = QLabel()
        self.left_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.left_image.setMinimumSize(400, 400)
        left_layout.addWidget(self.left_image, 1)
        
        self.left_label = QLabel("Image A")
        self.left_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.left_label)
        
        # Right image
        self.right_frame = QFrame()
        self.right_frame.setFrameShape(QFrame.Shape.StyledPanel)
        right_layout = QVBoxLayout(self.right_frame)
        
        self.right_image = QLabel()
        self.right_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.right_image.setMinimumSize(400, 400)
        right_layout.addWidget(self.right_image, 1)
        
        self.right_label = QLabel("Image B")
        self.right_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.right_label)
        
        # Add to layout with splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.left_frame)
        splitter.addWidget(self.right_frame)
        splitter.setSizes([500, 500])  # Equal initial sizes
        
        layout.addWidget(splitter)
        
    def set_images(self, left_image_path: Path, right_image_path: Path) -> None:
        """
        Set the images to display.
        
        Args:
            left_image_path: Path to left image
            right_image_path: Path to right image
        """
        self.left_pixmap = self._load_image(left_image_path)
        self.left_image.setPixmap(self.left_pixmap)
        self.left_label.setText(f"Image A ({left_image_path.name})")
        
        self.right_pixmap = self._load_image(right_image_path)
        self.right_image.setPixmap(self.right_pixmap)
        self.right_label.setText(f"Image B ({right_image_path.name})")
    
    def _load_image(self, image_path: Path) -> QPixmap:
        """
        Load an image from path and resize it appropriately.
        
        Args:
            image_path: Path to the image
            
        Returns:
            QPixmap: Resized image pixmap
        """
        pixmap = QPixmap(str(image_path))
        
        # Calculate scaling to fit in the widget while preserving aspect ratio
        label_size = self.left_image.size()
        scaled_pixmap = pixmap.scaled(
            label_size, 
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        return scaled_pixmap