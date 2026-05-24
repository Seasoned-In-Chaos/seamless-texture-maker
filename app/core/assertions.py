"""
Shared type assertions for the processing pipeline.

These checks are only active when ``__debug__`` is True
(i.e. when Python is *not* run with ``-O`` or ``-OO``).
They guard against accidental dtype or shape mismatches
without adding any overhead in production.
"""
from __future__ import annotations

import numpy as np

__all__ = ["assert_float32", "assert_3d", "assert_2d"]


def assert_float32(arr: np.ndarray, name: str = "array") -> None:
    """Assert that *arr* is a float32 ndarray (debug builds only)."""
    if __debug__:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name}: expected ndarray, got {type(arr).__name__}")
        if arr.dtype != np.float32:
            raise TypeError(
                f"{name}: expected float32, got {arr.dtype}"
            )


def assert_3d(arr: np.ndarray, name: str = "array") -> None:
    """Assert that *arr* has 3 dimensions (H, W, C)."""
    if __debug__:
        if arr.ndim != 3:
            raise ValueError(f"{name}: expected 3-D (H,W,C), got shape {arr.shape}")


def assert_2d(arr: np.ndarray, name: str = "array") -> None:
    """Assert that *arr* has 2 dimensions (H, W)."""
    if __debug__:
        if arr.ndim != 2:
            raise ValueError(f"{name}: expected 2-D (H,W), got shape {arr.shape}")
