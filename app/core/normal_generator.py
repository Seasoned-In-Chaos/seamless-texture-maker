"""
Material map generation controls - PBR Pipeline
"""
import cv2
import numpy as np
from numba import jit

from .ao_generator import generate_ao_map

@jit(nopython=True, fastmath=True)
def compute_gradients_jit(height_map, strength=1.0):
    h, w = height_map.shape
    gradients = np.zeros((h, w, 2), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            prev_x = height_map[y, (x - 1 + w) % w]
            next_x = height_map[y, (x + 1) % w]
            prev_y = height_map[(y - 1 + h) % h, x]
            next_y = height_map[(y + 1) % h, x]
            dx = (next_x - prev_x) * strength
            dy = (next_y - prev_y) * strength
            gradients[y, x, 0] = dx
            gradients[y, x, 1] = dy
    return gradients

@jit(nopython=True, fastmath=True)
def gradients_to_normals_jit(gradients, invert_y=False):
    h, w = gradients.shape[:2]
    normals = np.zeros((h, w, 3), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            dx = gradients[y, x, 0]
            dy = gradients[y, x, 1]
            if invert_y: dy = -dy
            nx = -dx; ny = -dy; nz = 1.0
            mag = np.sqrt(nx*nx + ny*ny + nz*nz)
            normals[y, x, 0] = nx / mag
            normals[y, x, 1] = ny / mag
            normals[y, x, 2] = nz / mag
    return normals

class NormalGenerator:
    @staticmethod
    def process(image, **params):
        """Processes and returns a specific PBR map based on parameters."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        else:
            gray = image.astype(np.float32) / 255.0

        def apply_contrast(src, mode):
            mode = (mode or "balanced").lower()
            out = src.astype(np.float32)
            if mode == "auto":
                lo, hi = np.percentile(out, (2.0, 98.0))
                if hi > lo:
                    out = (out - lo) / (hi - lo)
            elif mode == "soft":
                out = 0.5 + (out - 0.5) * 0.65
            elif mode == "sharp":
                out = 0.5 + (out - 0.5) * 1.65
            return np.clip(out, 0.0, 1.0)

        # Determine which map we are currently adjusting (based on params from MaterialControlPanel)
        # This is a bit tricky because the panel sends ALL params at once.
        # We'll generate all maps and return a dict or just handle them individually.
        
        # 1. NORMAL
        ni = params.get("normal_intensity", 0.5) * 5.0
        ns = params.get("normal_smooth", 0.3)
        nd = params.get("normal_detail", 0.4)
        
        normal_height = 1.0 - gray if params.get("normal_invert_height") else gray
        normal_height = apply_contrast(normal_height, params.get("normal_contrast", "balanced"))
        
        # Simple height for normal
        h_map = cv2.GaussianBlur(normal_height, (0, 0), sigmaX=0.15 + ns * 10.0)
        h_map = gray + (gray - h_map) * nd
        h_map = np.clip(h_map, 0, 1)
        
        grads = compute_gradients_jit(h_map, ni)
        normals_raw = gradients_to_normals_jit(grads, params.get("normal_format") == "directx")
        if params.get("normal_map_type") == "bump":
            normal_img = cv2.cvtColor((h_map * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        else:
            normal_img = np.zeros((h_map.shape[0], h_map.shape[1], 3), dtype=np.uint8)
            normal_img[..., 0] = np.clip((normals_raw[..., 2] * 0.5 + 0.5) * 255, 0, 255)
            normal_img[..., 1] = np.clip((normals_raw[..., 1] * 0.5 + 0.5) * 255, 0, 255)
            normal_img[..., 2] = np.clip((normals_raw[..., 0] * 0.5 + 0.5) * 255, 0, 255)

        # 2. ROUGHNESS
        ri = params.get("rough_intensity", 0.5)
        rc = params.get("rough_contrast", 0.0)
        rough = gray.copy()
        if params.get("rough_invert"): rough = 1.0 - rough
        rough = np.clip(0.5 + (rough - 0.5) * (1.0 + rc * 2.0), 0, 1)
        rough = np.clip(rough * (ri * 2.0), 0, 1)
        rough_img = cv2.cvtColor((rough * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # 3. METALLIC
        mi = params.get("metal_intensity", 0.0)
        me = params.get("metal_edge", 0.2)
        metal = np.zeros_like(gray)
        if mi > 0:
            metal = np.clip(gray * (mi * 2.0), 0, 1)
            if me > 0:
                sigma = 0.35 + me * 8.0
                metal = cv2.GaussianBlur(metal, (0, 0), sigmaX=sigma)
                metal = np.clip(metal, 0, 1)
        metal_img = cv2.cvtColor((metal * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # 4. AO
        ai = params.get("ao_intensity", 0.5)
        aspread = params.get("ao_spread", 0.3)
        ao_source = gray
        if aspread > 0:
            ao_source = cv2.GaussianBlur(gray, (0, 0), sigmaX=0.75 + aspread * 14.0)
        ao = 1.0 - (ao_source * ai)
        ao_img = cv2.cvtColor((ao * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # 5. HEIGHT
        hi = params.get("height_depth", 0.5)
        hs = params.get("height_smooth", 0.1)
        height_source = 1.0 - gray if params.get("height_invert") else gray
        height_source = apply_contrast(height_source, params.get("height_contrast", "balanced"))
        if hs > 0:
            height_source = cv2.GaussianBlur(height_source, (0, 0), sigmaX=0.35 + hs * 10.0)
        height = np.clip(height_source * (hi * 2.0), 0, 1)
        height_img = cv2.cvtColor((height * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        displacement_strength = params.get("displacement_strength", 0.2)
        displacement = np.clip(height * (0.25 + displacement_strength * 1.75), 0, 1)
        displacement_img = cv2.cvtColor((displacement * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # 6. OPACITY
        ot = params.get("alpha_threshold", 1.0)
        aso = params.get("alpha_softness", 0.0)
        threshold = 1.0 - ot
        if aso > 0:
            width = max(0.01, aso * 0.45)
            opacity = np.clip((gray - threshold + width * 0.5) / width, 0.0, 1.0).astype(np.float32)
        else:
            opacity = np.where(gray > threshold, 1.0, 0.0).astype(np.float32)
        opacity_img = cv2.cvtColor((opacity * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # 7. EMISSIVE
        ei = params.get("glow_intensity", 0.0)
        tint_name = params.get("glow_tint", "white")
        tint_bgr = {
            "white": np.array([1.0, 1.0, 1.0], dtype=np.float32),
            "warm": np.array([0.55, 0.82, 1.0], dtype=np.float32),
            "cool": np.array([1.0, 0.78, 0.48], dtype=np.float32),
            "custom": np.array([0.95, 0.55, 1.0], dtype=np.float32),
        }.get(tint_name, np.array([1.0, 1.0, 1.0], dtype=np.float32))
        emissive = np.clip(gray * ei, 0, 1)
        emissive_img = np.clip(emissive[..., None] * tint_bgr * 255.0, 0, 255).astype(np.uint8)

        return {
            "Normal": normal_img,
            "Roughness": rough_img,
            "Metallic": metal_img,
            "AO": ao_img,
            "Height": height_img,
            "Displacement": displacement_img,
            "Opacity": opacity_img,
            "Emissive": emissive_img
        }
