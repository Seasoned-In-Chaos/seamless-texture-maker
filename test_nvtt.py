import cv2
import numpy as np
from app.core.normal_generator import NormalGenerator

img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

params = {
    "height_source": "red",
    "filter_scale": "7x7",
    "normal_filter": "scharr",
    "normal_wrap": False,
    "normal_invert_x": True,
    "normal_invert_y": True,
    "normal_normalize": True,
    "normal_min_z": 0.35,
    "normal_scale": 1.2,
}

res = NormalGenerator.process(img, use_cache=False, **params)
normal = res["Normal"]

print(f"Normal map shape: {normal.shape}, dtype: {normal.dtype}")
print(f"Min: {normal.min()}, Max: {normal.max()}")

if normal.dtype != np.uint16:
    print("FAILED: Not uint16!")
else:
    print("SUCCESS: 16-bit normal map generated with NVTT parameters!")
