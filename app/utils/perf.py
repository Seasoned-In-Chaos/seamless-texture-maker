"""
Performance timing utilities for SEAMS.

Provides PerfTimer context manager, @timed decorator, and a
module-level PERF_LOG dict that accumulates timing data across
the entire processing pipeline.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, Optional

logger = logging.getLogger("seams.perf")

_lock = threading.Lock()
PERF_LOG: Dict[str, float] = {}


class PerfTimer:
    """Context manager that records elapsed time and logs it.

    Usage::

        with PerfTimer("edge_blend"):
            result = blend(...)
        # logs: "edge_blend: 12.3ms"
        # PERF_LOG["edge_blend"] == 12.3
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._t0: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "PerfTimer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0
        logger.debug("%s: %.1fms", self.name, self.elapsed_ms)
        with _lock:
            PERF_LOG[self.name] = self.elapsed_ms


def timed(name: Optional[str] = None) -> Callable:
    """Decorator that times a function call.

    Usage::

        @timed("compute_gradients")
        def compute_gradients(...):
            ...
    """

    def decorator(func: Callable) -> Callable:
        _name = name or func.__qualname__

        def wrapper(*args, **kwargs):
            with PerfTimer(_name):
                return func(*args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__qualname__ = func.__qualname__
        return wrapper

    return decorator


def get_perf_summary() -> Dict[str, float]:
    """Return a copy of the accumulated performance log."""
    with _lock:
        return dict(PERF_LOG)


def reset_perf_log() -> None:
    """Clear all accumulated timing data."""
    with _lock:
        PERF_LOG.clear()


def log_gpu_stats() -> None:
    """Log GPU device info if CUDA is available."""
    try:
        import cv2
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if count > 0:
            logger.info("CUDA device count: %d", count)
        else:
            logger.info("No CUDA devices available")
    except Exception as exc:
        logger.debug("GPU stats unavailable: %s", exc)

    try:
        import psutil
        mem = psutil.virtual_memory()
        logger.info("System RAM: %.1f GB (used %.1f GB)",
                     mem.total / (1024**3), mem.used / (1024**3))
    except Exception:
        pass
