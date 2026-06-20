import time
import cv2
import numpy as np

def compute_scharr_slopes(h_map):
    # Pad for seamless wrap
    padded = cv2.copyMakeBorder(h_map, 1, 1, 1, 1, cv2.BORDER_WRAP)
    # Scharr scale: the default cv2.Scharr computes gradient without dividing by 32.
    # The Scharr operator is [3, 10, 3], sum = 16, derivative is [1, 0, -1], diff = 2.
    # Overall scale factor for a true derivative is 1/32.
    scale = 1.0 / 32.0
    gx = cv2.Scharr(padded, cv2.CV_32F, 1, 0, scale=scale)
    gy = cv2.Scharr(padded, cv2.CV_32F, 0, 1, scale=scale)
    # Crop back
    return gx[1:-1, 1:-1], gy[1:-1, 1:-1]

def slope_to_normal(gx, gy, strength=1.0):
    gx = gx * strength
    gy = gy * strength
    # nx = -gx because positive gx means h(x+1) > h(x-1)
    mag = np.sqrt(gx**2 + gy**2 + 1.0)
    nx = -gx / mag
    ny = -gy / mag
    nz = 1.0 / mag
    
    # Pack to 16-bit BGR: Z is Blue, Y is Green, X is Red
    normals = np.stack([nz, ny, nx], axis=-1)
    normals = normals * 0.5 + 0.5
    return np.clip(normals * 65535.0, 0, 65535).astype(np.uint16)

h_map = np.random.rand(2048, 2048).astype(np.float32)

t0 = time.time()
for _ in range(5):
    # Multi-scale slope extraction
    gxs = []
    gys = []
    for sigma in [0.0, 1.0, 2.0, 4.0]:
        if sigma > 0:
            # We must use BORDER_WRAP for seamless blurring!
            blurred = cv2.copyMakeBorder(h_map, 10, 10, 10, 10, cv2.BORDER_WRAP)
            blurred = cv2.GaussianBlur(blurred, (0, 0), sigmaX=sigma)
            blurred = blurred[10:-10, 10:-10]
        else:
            blurred = h_map
        gx, gy = compute_scharr_slopes(blurred)
        gxs.append(gx)
        gys.append(gy)
    
    # Blending slopes
    gx_total = gxs[0]*0.4 + gxs[1]*0.3 + gxs[2]*0.2 + gxs[3]*0.1
    gy_total = gys[0]*0.4 + gys[1]*0.3 + gys[2]*0.2 + gys[3]*0.1
    
    normal16 = slope_to_normal(gx_total, gy_total, 5.0)

t1 = time.time()
print(f"Time per 2k image: {(t1-t0)/5.0 * 1000.0:.2f} ms")
print(f"Normal map shape: {normal16.shape}, dtype: {normal16.dtype}")
