"""
Base Storage Interface
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from ..models import Episode, SemanticMemory


class BaseStorage(ABC):
    """Base storage interface that all storage implementations must follow"""
    
    def __init__(self, storage_path: str, data_type: str = ""):
        """
        Initialize storage
        
        Args:
            storage_path: Path for storage
            data_type: Type of data being stored
        """
        self.storage_path = storage_path
        self.data_type = data_type
        
        # Create data directory for the specific type
        from pathlib import Path
        import os
        
        if data_type:
            self.data_dir = Path(storage_path) / data_type
            os.makedirs(self.data_dir, exist_ok=True)
        else:
            self.data_dir = Path(storage_path)
    
    @abstractmethod
    def save(self, item: Any) -> str:
        """
        Save an item and return its ID
        
        Args:
            item: Item to save
            
        Returns:
            Item ID
        """
        pass
    
    @abstractmethod
    def load(self, item_id: str) -> Optional[Any]:
        """
        Load an item by ID
        
        Args:
            item_id: Item ID
            
        Returns:
            Item or None if not found
        """
        pass
    
    @abstractmethod
    def delete(self, item_id: str) -> bool:
        """
        Delete an item by ID
        
        Args:
            item_id: Item ID
            
        Returns:
            True if deleted successfully
        """
        pass
    
    @abstractmethod
    def list_user_items(self, user_id: str) -> List[Any]:
        """
        List all items for a user
        
        Args:
            user_id: User ID
            
        Returns:
            List of items
        """
        pass
    
    @abstractmethod
    def delete_user_data(self, user_id: str) -> bool:
        """
        Delete all data for a user
        
        Args:
            user_id: User ID
            
        Returns:
            True if deleted successfully
        """
        pass
    
    @abstractmethod
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics for a user
        
        Args:
            user_id: User ID
            
        Returns:
            Statistics dictionary
        """
        pass
    
    def exists(self, item_id: str) -> bool:
        """
        Check if an item exists
        
        Args:
            item_id: Item ID
            
        Returns:
            True if exists
        """
        return self.load(item_id) is not None 