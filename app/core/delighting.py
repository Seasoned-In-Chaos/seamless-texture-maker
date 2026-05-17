"""
Delighting and Flattening logic for textures.
Removes directional lighting gradients and preserves high-frequency details.
"""
import cv2
import numpy as np

def delight_image(image, strength=0.5, flatness=0.0):
    """
    Remove lighting gradients and shadows from the image.
    
    Args:
        image: Input BGR image (uint8)
        strength: Strength of delighting (0.0 to 1.0)
        flatness: Amount of color flattening (0.0 to 1.0)
        
    Returns:
        Delighted BGR image
    """
    if strength < 0.01 and flatness < 0.01:
        return image.copy()
    
    # 1. Convert to LAB for luminance processing
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    
    # --- LUMINANCE DELIGHTING ---
    if strength > 0:
        # We use a very large blur to capture global lighting gradients
        h, w = l.shape
        sigma = max(h, w) * 0.1 # 10% of image size
        # Ensure sigma is within reasonable bounds
        sigma = np.clip(sigma, 10, 500)
        
        # Low frequency lighting
        k_size = int(sigma * 3)
        if k_size % 2 == 0: k_size += 1
        low_freq = cv2.GaussianBlur(l, (k_size, k_size), sigma)
        
        # The average luminance we want to maintain
        mean_l = np.mean(l)
        
        # High pass: detail = L - low_freq
        # Normalized: result = L / low_freq * mean_l
        # To avoid division by zero
        low_freq = np.clip(low_freq, 1, 255)
        
        # Apply delighting strength
        delighted_l = (l / low_freq) * mean_l
        
        # Blend with original based on strength
        l = l * (1.0 - strength) + delighted_l * strength
        
        # Keep within bounds
        l = np.clip(l, 0, 255)
        
    # --- COLOR FLATTENING ---
    if flatness > 0:
        # Reduce variance in A and B channels towards their mean
        mean_a = np.mean(a)
        mean_b = np.mean(b)
        
        a = a * (1.0 - flatness) + mean_a * flatness
        b = b * (1.0 - flatness) + mean_b * flatness
        
    # Recombine and convert back
    lab_processed = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab_processed.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    return result
