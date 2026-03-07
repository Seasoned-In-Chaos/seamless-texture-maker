"""
Normal and Bump Map Generator - Refactored Pipeline
Implements proper frequency separation and Photoshop-quality behavior.
"""
import cv2
import numpy as np
from numba import jit

# ============================================================================
# FREQUENCY SEPARATION
# ============================================================================

def separate_frequencies(height_map, sigma=10.0):
    """
    Separate height map into low and high frequency components.
    Uses bilateral filter for edge-aware separation.
    """
    # Low frequency: bilateral filter preserves edges
    low_freq = cv2.bilateralFilter(
        (height_map * 255).astype(np.uint8),
        d=0,
        sigmaColor=50,
        sigmaSpace=sigma
    ).astype(np.float32) / 255.0
    
    # High frequency: residual detail
    high_freq = height_map - low_freq
    
    return low_freq, high_freq


def recombine_frequencies(low_freq, high_freq, detail_scale=1.0):
    """
    Recombine frequency components with detail scale control.
    detail_scale: 0 = pure low freq, 1 = full detail
    """
    return low_freq + (high_freq * detail_scale)


# ============================================================================
# EDGE-AWARE SMOOTHING
# ============================================================================

def edge_aware_smooth(image, smoothness=0.0):
    """
    Apply edge-aware smoothing using bilateral filter.
    smoothness: 0 = no smoothing, 1 = maximum smoothing
    """
    if smoothness < 0.01:
        return image
    
    # Scale smoothness to bilateral filter parameters
    sigma_color = 20 + (smoothness * 80)  # 20-100
    sigma_space = 2 + (smoothness * 18)    # 2-20
    
    smoothed = cv2.bilateralFilter(
        (image * 255).astype(np.uint8),
        d=0,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space
    ).astype(np.float32) / 255.0
    
    return smoothed


# ============================================================================
# HEIGHT MAP GENERATION
# ============================================================================

def generate_height_map(image, invert=False, contrast_mode='balanced'):
    """
    Generate true height map from image using luminance conversion
    and automatic contrast normalization.
    """
    # 1. Luminance conversion (proper perceptual weighting)
    if len(image.shape) == 3:
        # Use ITU-R BT.709 luma coefficients
        height = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    else:
        height = image.astype(np.float32) / 255.0
    
    # 2. Invert Height (EARLY - before processing)
    # Applied to raw height data to ensure physical surface inversion
    if invert:
        height = 1.0 - height

    # 3. Auto contrast normalization
    height = apply_contrast(height, contrast_mode)
    
    # 4. Midtone emphasis for better depth perception
    # Apply a subtle S-curve
    height = apply_midtone_emphasis(height)
    
    return height


def apply_midtone_emphasis(height):
    """
    Apply subtle S-curve for better midtone contrast.
    """
    # Simple power curve that emphasizes midtones
    # Values near 0.5 get more contrast
    return np.power(height, 0.9)


def apply_contrast(height, mode):
    """Apply contrast adjustment to height map."""
    if mode == 'soft':
        return (height - 0.5) * 0.7 + 0.5
    elif mode == 'sharp':
        return np.clip((height - 0.5) * 1.5 + 0.5, 0, 1)
    elif mode == 'auto':
        # Auto levels (normalize)
        min_val = np.min(height)
        max_val = np.max(height)
        if max_val > min_val:
            return (height - min_val) / (max_val - min_val)
        return height
    return height  # balanced/default


# ============================================================================
# GRADIENT COMPUTATION (JIT OPTIMIZED)
# ============================================================================

@jit(nopython=True, fastmath=True)
def compute_gradients_jit(height_map, strength=1.0):
    """
    Compute gradients from height map with wrap-around sampling.
    Returns unnormalized gradient vectors.
    """
    h, w = height_map.shape
    gradients = np.zeros((h, w, 2), dtype=np.float32)
    
    for y in range(h):
        for x in range(w):
            # Wrap-around sampling for seamlessness
            prev_x = height_map[y, (x - 1 + w) % w]
            next_x = height_map[y, (x + 1) % w]
            prev_y = height_map[(y - 1 + h) % h, x]
            next_y = height_map[(y + 1) % h, x]
            
            # Central difference
            dx = (next_x - prev_x) * strength
            dy = (next_y - prev_y) * strength
            
            gradients[y, x, 0] = dx
            gradients[y, x, 1] = dy
    
    return gradients


@jit(nopython=True, fastmath=True)
def gradients_to_normals_jit(gradients, invert_y=False):
    """
    Convert gradient vectors to normalized normal vectors.
    """
    h, w = gradients.shape[:2]
    normals = np.zeros((h, w, 3), dtype=np.float32)
    
    for y in range(h):
        for x in range(w):
            dx = gradients[y, x, 0]
            dy = gradients[y, x, 1]
            
            if invert_y:
                dy = -dy
            
            # Normal vector
            nx = -dx
            ny = -dy
            nz = 1.0
            
            # Normalize
            mag = np.sqrt(nx*nx + ny*ny + nz*nz)
            normals[y, x, 0] = nx / mag
            normals[y, x, 1] = ny / mag
            normals[y, x, 2] = nz / mag
    
    return normals


# ============================================================================
# MAIN GENERATOR CLASS
# ============================================================================

class NormalGenerator:
    """Orchestrates normal and bump map generation."""
    
    @staticmethod
    def process(image, intensity=1.0, detail_scale=1.0, smoothness=0.0,
                invert_height=False, format='opengl', contrast_mode='balanced',
                map_type='normal', height_intensity=1.0):
        """
        Generate normal map or bump map with proper frequency separation.
        
        Pipeline:
        1. Generate base height map (with optional inversion)
        2. Separate into low/high frequencies
        3. Apply edge-aware smoothing to low freq
        4. Recombine with detail scale
        5a. Bump mode: Output pure height field (true height-field representation)
        5b. Normal mode: Compute gradients, scale by intensity, normalize
        
        Args:
            intensity: Controls both gradient strength (Normal) and height scale (Bump)
            height_intensity: Unused (kept for API compatibility)
        
        Note: Both modes output from the SAME height field for consistency.
              Inversion is applied at height-field level, before processing.
        """
        # Step 1: Generate base height map
        height = generate_height_map(image, invert_height, contrast_mode)
        
        # Step 2: Frequency separation
        low_freq, high_freq = separate_frequencies(height, sigma=10.0)
        
        # Step 3: Edge-aware smoothing on LOW frequency only
        low_freq_smoothed = edge_aware_smooth(low_freq, smoothness)
        
        # Step 4: Recombine with detail scale control
        final_height = recombine_frequencies(low_freq_smoothed, high_freq, detail_scale)
        
        # Clamp to valid range
        final_height = np.clip(final_height, 0, 1)
        
        # BUMP MAP MODE: Output height field with intensity scaling
        if map_type == 'bump':
            # Apply intensity (shared slider)
            # Use intensity arg, treating 1.0 as neutral (no change)
            # intensity < 1.0: compress height
            # intensity > 1.0: exaggerate height
            adjusted_height = 0.5 + (final_height - 0.5) * intensity
            adjusted_height = np.clip(adjusted_height, 0, 1)
            
            h, w = adjusted_height.shape
            bump_map = np.zeros((h, w, 3), dtype=np.uint8)
            gray_uint8 = (adjusted_height * 255).astype(np.uint8)
            bump_map[..., 0] = gray_uint8
            bump_map[..., 1] = gray_uint8
            bump_map[..., 2] = gray_uint8
            return bump_map, None
        
        # NORMAL MAP MODE: Continue with gradient computation
        
        # Step 5: Compute gradients (wrap-around for seamlessness)
        # Intensity is applied HERE, not to height map
        gradient_strength = intensity * 3.0  # Scale for visibility
        gradients = compute_gradients_jit(final_height, gradient_strength)
        
        # Step 6: Convert gradients to normalized normals
        invert_y = (format.lower() == 'directx')
        normals_raw = gradients_to_normals_jit(gradients, invert_y)
        
        # Step 7: Pack into BGR image
        h, w = final_height.shape
        normal_map = np.zeros((h, w, 3), dtype=np.uint8)
        normal_map[..., 0] = np.clip((normals_raw[..., 2] * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)  # B = Z
        normal_map[..., 1] = np.clip((normals_raw[..., 1] * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)  # G = Y
        normal_map[..., 2] = np.clip((normals_raw[..., 0] * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)  # R = X
        
        return normal_map, normals_raw
