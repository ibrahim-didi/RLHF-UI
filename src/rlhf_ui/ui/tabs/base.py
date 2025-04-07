"""
Base class for UI tabs.

This module provides a base class that standardizes the interface for all UI tabs.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional

import gradio as gr


class BaseTab(ABC):
    """
    Base class for all UI tabs in the RLHF UI application.
    
    This class defines the standard interface that all tabs should implement.
    Each tab is responsible for creating its own UI components and event handlers.
    """
    
    @abstractmethod
    def build(self, app_instance: Any) -> None:
        """
        Build the tab's UI components.
        
        Args:
            app_instance: The parent application instance
        """
        pass
    
    @abstractmethod
    def register_event_handlers(self) -> None:
        """
        Register event handlers for the tab's components.
        
        This method should be called after all UI components are created.
        """
        pass
    
    def get_name(self) -> str:
        """
        Get the display name for this tab.
        
        Returns:
            str: Display name for the tab
        """
        # Default implementation returns the class name without "Tab" suffix
        tab_name = self.__class__.__name__
        if tab_name.endswith("Tab"):
            tab_name = tab_name[:-3]
        
        # Insert spaces before capital letters for better readability
        import re
        tab_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', tab_name)
        
        return tab_name


class TabRegistry:
    """
    Registry for UI tabs.
    
    This class manages the collection of tabs in the application.
    """
    
    def __init__(self):
        """Initialize the tab registry."""
        self.tabs: Dict[str, BaseTab] = {}
    
    def register(self, tab_id: str, tab: BaseTab) -> None:
        """
        Register a tab with the registry.
        
        Args:
            tab_id: Unique identifier for the tab
            tab: Tab instance to register
        """
        self.tabs[tab_id] = tab
    
    def get(self, tab_id: str) -> Optional[BaseTab]:
        """
        Get a tab by its ID.
        
        Args:
            tab_id: Unique identifier for the tab
            
        Returns:
            BaseTab: The requested tab or None if not found
        """
        return self.tabs.get(tab_id)
    
    def get_all(self) -> Dict[str, BaseTab]:
        """
        Get all registered tabs.
        
        Returns:
            Dict[str, BaseTab]: All registered tabs
        """
        return self.tabs 