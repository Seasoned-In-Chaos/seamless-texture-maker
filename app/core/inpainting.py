"""
Smart inpainting for seam removal in textures.
Uses OpenCV's inpainting algorithms to seamlessly fill seam areas.
"""
import numpy as np
import cv2


def create_seam_detection_mask(image, threshold=30, seam_width=20):
    """
    Detect seams in the image based on gradient discontinuities.
    
    Args:
        image: Input image (BGR or grayscale)
        threshold: Gradient threshold for seam detection
        seam_width: Width to expand detected seams
    
    Returns:
        Binary mask of detected seams
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Compute gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Threshold to find strong edges (potential seams)
    seam_mask = (magnitude > threshold).astype(np.uint8) * 255
    
    # Dilate to expand seam regions
    kernel = np.ones((seam_width, seam_width), np.uint8)
    seam_mask = cv2.dilate(seam_mask, kernel, iterations=1)
    
    return seam_mask


def inpaint_seams(image, mask, method='telea', radius=5):
    """
    Inpaint the seam regions using OpenCV's inpainting.
    
    Args:
        image: Input image (BGR)
        mask: Binary mask where seams are marked (255)
        method: 'telea' or 'ns' (Navier-Stokes)
        radius: Inpainting radius
    
    Returns:
        Inpainted image
    """
    if method == 'telea':
        flags = cv2.INPAINT_TELEA
    else:
        flags = cv2.INPAINT_NS
    
    # Ensure mask is uint8
    mask = mask.astype(np.uint8)
    
    # Apply inpainting
    result = cv2.inpaint(image, mask, radius, flags)
    
    return result


def smart_seam_inpaint(image, seam_width=30, detail_preservation=0.5, method='telea'):
    """
    Smart inpainting that preserves texture details while removing seams.
    
    Args:
        image: Input image (offset so seams are at center)
        seam_width: Width of the seam region to inpaint
        detail_preservation: How much detail to preserve (0.0-1.0)
        method: Inpainting method ('telea' or 'ns')
    
    Returns:
        Image with inpainted seams
    """
    h, w = image.shape[:2]
    
    # Create cross mask at center (where seams are after offset)
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Adjust seam width based on detail preservation
    # Higher detail preservation = narrower inpaint region
    adjusted_width = int(seam_width * (1.0 - detail_preservation * 0.5))
    adjusted_width = max(2, adjusted_width)
    
    # Vertical seam at center
    cx = w // 2
    x1 = max(0, cx - adjusted_width // 2)
    x2 = min(w, cx + adjusted_width // 2)
    mask[:, x1:x2] = 255
    
    # Horizontal seam at center
    cy = h // 2
    y1 = max(0, cy - adjusted_width // 2)
    y2 = min(h, cy + adjusted_width // 2)
    mask[y1:y2, :] = 255
    
    # Inpainting radius based on seam width
    radius = max(3, adjusted_width // 2)
    
    # Apply inpainting
    result = inpaint_seams(image, mask, method, radius)
    
    # Blend with original to preserve some details if needed
    if detail_preservation > 0:
        # Create soft mask for blending
        soft_mask = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 0)
        soft_mask = soft_mask / 255.0
        
        # Reduce blend based on detail preservation
        blend_factor = soft_mask * (1.0 - detail_preservation * 0.3)
        
        if len(image.shape) == 3:
            blend_factor = blend_factor[:, :, np.newaxis]
        
        result = (result * (1 - blend_factor) + image * blend_factor).astype(image.dtype)
    
    return result


def multi_scale_inpaint(image, mask, scales=[1.0, 0.5, 0.25]):
    """
    Multi-scale inpainting for better texture preservation.
    
    Args:
        image: Input image
        mask: Inpainting mask
        scales: List of scales to process
    
    Returns:
        Inpainted image
    """
    h, w = image.shape[:2]
    result = image.copy()
    
    for scale in scales:
        if scale < 1.0:
            # Downscale
            new_w = int(w * scale)
            new_h = int(h * scale)
            scaled_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            scaled_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            scaled_img = image
            scaled_mask = mask
        
        # Inpaint at this scale
        radius = max(3, int(5 * scale))
        inpainted = cv2.inpaint(scaled_img, scaled_mask, radius, cv2.INPAINT_TELEA)
        
        if scale < 1.0:
            # Upscale back
            inpainted = cv2.resize(inpainted, (w, h), interpolation=cv2.INTER_CUBIC)
        
        # Blend with result
        weight = scale
        result = cv2.addWeighted(result, 1 - weight * 0.3, inpainted, weight * 0.3, 0)
    
    return result
