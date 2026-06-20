import cv2
import numpy as np
from app.core.normal_generator import NormalGenerator

img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

params = {
    "height_source": "red",
    "filter_scale": "dudv",
    "normal_filter": "sobel",
    "normal_wrap": True,
    "normal_invert_x": False,
    "normal_invert_y": False,
    "normal_normalize": True,
    "normal_min_z": 0.0,
    "normal_scale": 2.0,
}

res = NormalGenerator.process(img, use_cache=False, **params)
normal = res["Normal"]

print(f"Normal map shape: {normal.shape}, dtype: {normal.dtype}")
print(f"Min: {normal.min()}, Max: {normal.max()}")
print(f"Mean: {normal.mean()}")
