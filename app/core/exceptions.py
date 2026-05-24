"""Custom exception hierarchy for SEAMS."""
from __future__ import annotations

__all__ = ["SeamsError", "ImageLoadError", "ProcessingError", "GPUError", "CacheError"]


class SeamsError(Exception):
    """Base exception for all SEAMS errors."""


class ImageLoadError(SeamsError):
    """Raised when an image cannot be loaded or decoded."""


class ProcessingError(SeamsError):
    """Raised when texture processing fails."""


class GPUError(SeamsError):
    """Raised when a GPU operation fails after fallback."""


class CacheError(SeamsError):
    """Raised when a cache operation fails."""
