"""
Offset mapping for seamless texture generation.
Implements horizontal and vertical texture wrapping.
"""
import numpy as np


def offset_image(image, offset_x=0.5, offset_y=0.5):
    """
    Offset an image by a fraction of its dimensions.
    This wraps the texture to expose seams at the center.
    
    Args:
        image: Input image (numpy array, HxWxC or HxW)
        offset_x: Horizontal offset as fraction (0.0-1.0), default 0.5
        offset_y: Vertical offset as fraction (0.0-1.0), default 0.5
    
    Returns:
        Offset image with wrapped edges
    """
    h, w = image.shape[:2]
    
    # Calculate pixel offsets
    shift_x = int(w * offset_x)
    shift_y = int(h * offset_y)
    
    # Use numpy roll for efficient circular shift
    result = np.roll(image, shift_x, axis=1)  # Horizontal
    result = np.roll(result, shift_y, axis=0)  # Vertical
    
    return result


def reverse_offset(image, offset_x=0.5, offset_y=0.5):
    """
    Reverse the offset operation to restore original positioning.
    
    Args:
        image: Offset image (numpy array)
        offset_x: Original horizontal offset fraction
        offset_y: Original vertical offset fraction
    
    Returns:
        Image with offset reversed
    """
    h, w = image.shape[:2]
    
    shift_x = int(w * offset_x)
    shift_y = int(h * offset_y)
    
    # Reverse the shifts
    result = np.roll(image, -shift_y, axis=0)
    result = np.roll(result, -shift_x, axis=1)
    
    return result


def get_seam_mask(image, seam_width=50):
    """
    Create a mask highlighting the seam areas after offset.
    The seams appear at the center after a 0.5 offset.
    
    Args:
        image: Input image (used for dimensions)
        seam_width: Width of the seam region in pixels
    
    Returns:
        Binary mask where seam regions are 255
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Horizontal center seam
    center_x = w // 2
    x_start = max(0, center_x - seam_width // 2)
    x_end = min(w, center_x + seam_width // 2)
    mask[:, x_start:x_end] = 255
    
    # Vertical center seam
    center_y = h // 2
    y_start = max(0, center_y - seam_width // 2)
    y_end = min(h, center_y + seam_width // 2)
    mask[y_start:y_end, :] = 255
    
    return mask


def create_cross_mask(height, width, thickness=50):
    """
    Create a cross-shaped mask for the center seams.
    
    Args:
        height: Image height
        width: Image width
        thickness: Thickness of the cross arms
    
    Returns:
        Binary mask (uint8)
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Vertical bar
    x_center = width // 2
    x1 = max(0, x_center - thickness // 2)
    x2 = min(width, x_center + thickness // 2)
    mask[:, x1:x2] = 255
    
    # Horizontal bar
    y_center = height // 2
    y1 = max(0, y_center - thickness // 2)
    y2 = min(height, y_center + thickness // 2)
    mask[y1:y2, :] = 255
    
    return mask
