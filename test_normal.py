import sys
sys.path.insert(0, 'e:\\seamless-texture-maker')
import numpy as np
from app.core.normal_generator import compute_gradients_jit, _compute_normal

# Test compute_gradients_jit
height_map = np.random.rand(100, 100).astype(np.float32)
weights = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
grads = compute_gradients_jit(height_map, weights, 1.0)
print(f"JIT Grads shape: {grads.shape}, min/max: {grads.min():.3f} / {grads.max():.3f}")

# Test _compute_normal
params = {
    "normal_intensity": 0.5,
    "normal_smooth": 0.1,
    "normal_detail": 0.5,
    "normal_format": "opengl"
}
gray = np.random.rand(100, 100).astype(np.float32)
normal_img = _compute_normal(height_map, gray, params)
print(f"Normal map shape: {normal_img.shape}, dtype: {normal_img.dtype}")
print("Test completed successfully!")
