import cv2
import numpy as np
from app.core.normal_generator import NormalGenerator

img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

res1 = NormalGenerator.process(img, use_cache=False, normal_intensity=1.0, normal_smooth=0.0, normal_detail=0.5, normal_filter="scharr")
normal_scharr = res1["Normal"]

res2 = NormalGenerator.process(img, use_cache=False, normal_intensity=1.0, normal_smooth=0.0, normal_detail=0.5, normal_filter="sobel")
normal_sobel = res2["Normal"]

print(f"Scharr map shape: {normal_scharr.shape}, dtype: {normal_scharr.dtype}")
print(f"Sobel map shape: {normal_sobel.shape}, dtype: {normal_sobel.dtype}")

if normal_scharr.dtype != np.uint16 or normal_sobel.dtype != np.uint16:
    print("FAILED: Not uint16!")
else:
    print("SUCCESS: 16-bit normal maps generated with both filters!")
