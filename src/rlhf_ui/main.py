# src/rlhf_ui/main.py
#!/usr/bin/env python
"""
Main entry point for the RLHF UI application.
"""

import sys
import logging
from PyQt6.QtWidgets import QApplication
from rlhf_ui.ui.app import RLHFApplication
from rlhf_ui.config import load_config

def setup_logging():
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("rlhf_ui.log")
        ]
    )

def main():
    """Start the RLHF UI application."""
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting RLHF UI application")
    
    # Load configuration
    config = load_config()
    
    # Start Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("RLHF UI")
    app.setOrganizationName("Your Organization")
    
    # Create and show main window
    main_window = RLHFApplication(config)
    main_window.show()
    
    # Start event loop
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())