import cv2
import numpy as np

def generate_ao_map(height_map, normal_map=None, radius=16, strength=1.0, contrast=1.0):
    """
    Approximates an Ambient Occlusion (AO) map from a height map (and optionally a normal map).
    
    Args:
        height_map (np.ndarray): Single-channel grayscale height map (0-255).
        normal_map (np.ndarray, optional): 3-channel normal map (BGR/RGB).
        radius (int): Blur radius to approximate global illumination.
        strength (float): Intensity of the AO shadows.
        contrast (float): Contrast multiplier.
        
    Returns:
        np.ndarray: Grayscale AO map (0-255).
    """
    if len(height_map.shape) == 3:
        height_map = cv2.cvtColor(height_map, cv2.COLOR_BGR2GRAY)
        
    # Ensure height_map is float32 0.0-1.0
    h = height_map.astype(np.float32) / 255.0
    
    # 1. Height-based occlusion (Unsharp Mask approach)
    # We blur the height map at multiple scales to simulate different ray lengths
    occlusion = np.zeros_like(h)
    
    # Multi-scale approach for better depth
    scales = [radius // 4, radius // 2, radius]
    weights = [0.5, 0.3, 0.2]
    
    for scale, weight in zip(scales, weights):
        if scale < 1:
            scale = 1
        ksize = scale if scale % 2 == 1 else scale + 1
        blurred_h = cv2.GaussianBlur(h, (ksize, ksize), 0)
        
        # Diff = blurred - original. Positive where original is deeper than average.
        diff = np.clip(blurred_h - h, 0.0, 1.0)
        occlusion += diff * weight
    
    # Scale occlusion by strength
    occlusion = occlusion * (strength * 8.0)
    
    # 2. Normal-based occlusion (optional)
    if normal_map is not None:
        # Assuming BGR or RGB format. The Z channel is the 3rd channel (index 2)
        nz = (normal_map[..., 2].astype(np.float32) / 255.0) * 2.0 - 1.0
        
        # Steeper surfaces (nz near 0.0) have more potential occlusion
        steepness = np.clip(1.0 - nz, 0.0, 1.0)
        
        # Combine normal steepness with height crevice diff
        occlusion = occlusion + (steepness * occlusion * strength * 0.5)
    
    # Apply contrast using a power curve
    occlusion = np.power(np.clip(occlusion, 0.0, 1.0), 1.0 / contrast)
    
    # AO is the inverse of occlusion (1.0 = fully lit, 0.0 = fully occluded)
    ao = 1.0 - occlusion
    
    return (ao * 255.0).astype(np.uint8)
