"""High-quality CPU texture filtering for the realtime viewport."""

from __future__ import annotations

import cv2
import numpy as np


def _to_rgba_u8(image: np.ndarray) -> np.ndarray:
    """Convert an OpenCV image to owned RGBA uint8 pixels."""
    if image.dtype == np.uint16:
        image = (image / 257.0).round().astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)


def _resize_area(image: np.ndarray, width: int, height: int) -> np.ndarray:
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    if image.ndim == 3 and resized.ndim == 2:
        resized = resized[..., None]
    return resized


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * (rgb ** (1.0 / 2.4)) - 0.055)


def _downsample_color(rgba: np.ndarray, width: int, height: int) -> np.ndarray:
    """Downsample sRGB color using linear-light premultiplied-alpha filtering."""
    rgb = _srgb_to_linear(rgba[..., :3].astype(np.float32) / 255.0)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    premultiplied = rgb * alpha
    filtered_rgb = _resize_area(premultiplied, width, height)
    filtered_alpha = _resize_area(alpha, width, height)
    rgb = filtered_rgb / np.maximum(filtered_alpha, 1e-6)
    out = np.concatenate((_linear_to_srgb(rgb), filtered_alpha), axis=2)
    return np.rint(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def _downsample_normal(rgba: np.ndarray, width: int, height: int) -> np.ndarray:
    """Downsample encoded normals by averaging slopes, then renormalizing."""
    encoded = rgba[..., :3].astype(np.float32) / 127.5 - 1.0
    nz = np.maximum(encoded[..., 2], 1e-4)
    slopes = np.stack((encoded[..., 0] / nz, encoded[..., 1] / nz), axis=2)
    slopes = _resize_area(slopes, width, height)
    normal = np.concatenate((slopes, np.ones((height, width, 1), dtype=np.float32)), axis=2)
    normal /= np.maximum(np.linalg.norm(normal, axis=2, keepdims=True), 1e-6)
    out_rgb = np.rint(np.clip(normal * 127.5 + 127.5, 0.0, 255.0)).astype(np.uint8)
    alpha = _resize_area(rgba[..., 3:4], width, height)
    return np.concatenate((out_rgb, alpha), axis=2)


def generate_mipmaps(image: np.ndarray, map_name: str) -> list[np.ndarray]:
    """Build mip levels with map-aware filtering, including the base level."""
    rgba = _to_rgba_u8(image)
    levels = [rgba]
    is_normal = map_name.lower() == "normal"
    is_color = map_name.lower() in {"base color", "basecolor"}
    while rgba.shape[0] > 1 or rgba.shape[1] > 1:
        height = max(1, rgba.shape[0] // 2)
        width = max(1, rgba.shape[1] // 2)
        if is_normal:
            rgba = _downsample_normal(rgba, width, height)
        elif is_color:
            rgba = _downsample_color(rgba, width, height)
        else:
            rgba = _resize_area(rgba, width, height)
        levels.append(np.ascontiguousarray(rgba))
    return levels
