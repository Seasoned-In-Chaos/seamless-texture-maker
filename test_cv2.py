import cv2
import numpy as np

img = np.zeros((10, 10), dtype=np.float32)
try:
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, scale=0.125)
    print("Sobel kwargs OK")
except Exception as e:
    print("Sobel kwargs ERROR:", e)

try:
    gx = cv2.Scharr(img, cv2.CV_32F, 1, 0, scale=0.125)
    print("Scharr kwargs OK")
except Exception as e:
    print("Scharr kwargs ERROR:", e)
