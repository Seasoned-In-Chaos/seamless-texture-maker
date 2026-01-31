"""
Result caching system for instant preview updates.
"""
import hashlib
import pickle
import numpy as np
from functools import lru_cache


class ResultCache:
    """Cache for processed texture results."""
    
    def __init__(self, max_size=50):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    def _hash_params(self, params):
        """Create hash key from parameters."""
        # Convert params dict to sorted tuple for consistent hashing
        param_str = str(sorted(params.items()))
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def get(self, params, image_hash=None):
        """
        Get cached result if available.
        
        Args:
            params: Processing parameters
            image_hash: Optional hash of source image
        
        Returns:
            Cached result or None
        """
        key = self._hash_params(params)
        if image_hash:
            key = f"{image_hash}_{key}"
        
        if key in self.cache:
            # Update access order (LRU)
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key].copy()
        
        return None
    
    def set(self, params, result, image_hash=None):
        """
        Store result in cache.
        
        Args:
            params: Processing parameters
            result: Processed image
            image_hash: Optional hash of source image
        """
        key = self._hash_params(params)
        if image_hash:
            key = f"{image_hash}_{key}"
        
        # Evict oldest if cache full
        if len(self.cache) >= self.max_size and key not in self.cache:
            if self.access_order:
                oldest = self.access_order.pop(0)
                del self.cache[oldest]
        
        self.cache[key] = result.copy()
        if key not in self.access_order:
            self.access_order.append(key)
    
    def clear(self):
        """Clear all cached results."""
        self.cache.clear()
        self.access_order.clear()
    
    def get_stats(self):
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'memory_mb': sum(r.nbytes for r in self.cache.values()) / (1024 * 1024)
        }


def hash_image(image):
    """
    Create fast hash of image for cache key.
    
    Args:
        image: NumPy array
    
    Returns:
        Hash string
    """
    # Sample pixels for speed (don't hash entire image)
    h, w = image.shape[:2]
    step = max(h // 16, w // 16, 1)
    sample = image[::step, ::step]
    
    # Hash shape + sample
    hash_data = f"{image.shape}_{sample.tobytes()}"
    return hashlib.md5(hash_data.encode()).hexdigest()[:8]
