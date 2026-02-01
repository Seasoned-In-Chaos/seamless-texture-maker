"""
Core logic for Normal and Bump map generation.
Inspired by professional texturing tools, optimized for speed.
"""
import cv2
import numpy as np
from numba import jit, prange

@jit(nopython=True, fastmath=True)
def _calculate_normals_jit(height_map, intensity, invert_y=False):
    """
    JIT-optimized normal map calculation.
    Input: Normalised grayscale height (0-1)
    """
    h, w = height_map.shape
    normals = np.zeros((h, w, 3), dtype=np.float32)
    
    # Scale factor for intensity
    strength = intensity * 0.1
    
    for y in range(h):
        for x in range(w):
            # Central difference with wrap-around
            prev_x = height_map[y, (x - 1 + w) % w]
            next_x = height_map[y, (x + 1) % w]
            prev_y = height_map[(y - 1 + h) % h, x]
            next_y = height_map[(y + 1) % h, x]
            
            # Gradient
            dx = (next_x - prev_x) * strength
            dy = (next_y - prev_y) * strength
            
            if invert_y:
                dy = -dy
                
            # Normal vector (nx, ny, nz)
            # Tangent space: x is dx, y is dy, z is 1.0
            nx = -dx
            ny = -dy
            nz = 1.0
            
            # Normalize
            mag = np.sqrt(nx*nx + ny*ny + nz*nz)
            normals[y, x, 0] = nx / mag
            normals[y, x, 1] = ny / mag
            normals[y, x, 2] = nz / mag
            
    return normals

@jit(nopython=True, parallel=True, fastmath=True)
def compute_lighting_jit(normals, light_pos, specular_power=32.0):
    """
    Computes real-time lighting for a normal map.
     normals: (H, W, 3) float32 in range [-1, 1]
     light_pos: (3,) light vector (normalized)
    """
    h, w = normals.shape[:2]
    shading = np.zeros((h, w), dtype=np.float32)
    
    for y in prange(h):
        for x in range(w):
            nx = normals[y, x, 0]
            ny = normals[y, x, 1]
            nz = normals[y, x, 2]
            
            # Diffuse: N dot L
            dot = nx * light_pos[0] + ny * light_pos[1] + nz * light_pos[2]
            diffuse = max(0.1, dot) # Soft ambient
            
            # Simple Blinn-Phong Specular
            # Halfway vector H approximation (view is straight down Z)
            # H = normalize(L + V). V = (0,0,1)
            hx = light_pos[0]
            hy = light_pos[1]
            hz = light_pos[2] + 1.0
            hmag = np.sqrt(hx*hx + hy*hy + hz*hz)
            
            if hmag > 0.001:
                ndoth = (nx * (hx/hmag) + ny * (hy/hmag) + nz * (hz/hmag))
                spec = np.power(max(0, ndoth), specular_power)
            else:
                spec = 0.0
                
            shading[y, x] = diffuse + spec * 0.5
            
    return np.clip(shading, 0, 1)

class NormalGenerator:
    """Orchestrates normal map generation process."""
    
    @staticmethod
    def process(image, intensity=1.0, detail_scale=1.0, smoothness=0.0, 
                invert_height=False, format='opengl', contrast_mode='balanced'):
        """
        Generate normal map from image.
        """
        # 1. Grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            gray = image.astype(np.float32) / 255.0
            
        if invert_height:
            gray = 1.0 - gray
            
        # 2. Contrast Enhancement
        gray = NormalGenerator._apply_contrast(gray, contrast_mode)
        
        # 3. Filtering
        # Smoothness: Global noise reduction
        if smoothness > 0.01:
            gray = cv2.GaussianBlur(gray, (0, 0), smoothness * 5.0)
            
        # Detail Scale: Fine (0) to Coarse (1.0)
        if detail_scale > 0.01:
            gray = cv2.GaussianBlur(gray, (0, 0), detail_scale * 10.0)
            
        # 4. Generate Normals
        invert_y = (format.lower() == 'directx')
        # Scale intensity to a more visible range
        effective_intensity = intensity * 5.0
        normals_raw = _calculate_normals_jit(gray, effective_intensity, invert_y)
        
        # 5. Pack into BGR Image (always 3 channels)
        h, w = gray.shape
        normal_map = np.zeros((h, w, 3), dtype=np.uint8)
        normal_map[..., 0] = np.clip((normals_raw[..., 2] * 0.5 + 0.5) * 255, 0, 255) # B = Z
        normal_map[..., 1] = np.clip((normals_raw[..., 1] * 0.5 + 0.5) * 255, 0, 255) # G = Y
        normal_map[..., 2] = np.clip((normals_raw[..., 0] * 0.5 + 0.5) * 255, 0, 255) # R = X
        
        return normal_map, normals_raw

    @staticmethod
    def _apply_contrast(gray, mode):
        if mode == 'soft':
            return (gray - 0.5) * 0.7 + 0.5
        elif mode == 'sharp':
            return np.clip((gray - 0.5) * 1.5 + 0.5, 0, 1)
        elif mode == 'auto':
            # Auto levels (normalize)
            min_val = np.min(gray)
            max_val = np.max(gray)
            if max_val > min_val:
                return (gray - min_val) / (max_val - min_val)
            return gray
        return gray # balanced/default

    @staticmethod
    def generate_preview(image, intensity=1.0, detail_scale=1.0, smoothness=0.0, 
                        invert_height=False, format='opengl', contrast_mode='balanced',
                        light_dir=(0.5, 0.5, 1.0)):
        """Generate a lit preview image."""
        # 1. Get normals
        _, normals = NormalGenerator.process(image, intensity, detail_scale, smoothness, 
                                           invert_height, format, contrast_mode)
        
        # 2. Normalize light dir
        ldir = np.array(light_dir, dtype=np.float32)
        ldir /= np.linalg.norm(ldir)
        
        # 3. Compute Shading
        shading = compute_lighting_jit(normals, ldir)
        
        # 4. Multiplicative blend with base color? 
        # User wants "Real-time lighting preview". Usually this means a shaded gray or basic tint.
        # Let's use a nice neutral material look.
        base_color = np.array([180, 180, 180], dtype=np.float32)
        shaded = (base_color * shading[..., np.newaxis]).astype(np.uint8)
        
        return shaded
