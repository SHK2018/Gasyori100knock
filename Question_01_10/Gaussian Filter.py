# -*- coding: utf-8 -*-
import cv2
import numpy as np


def gaussian_filter(img, kernel):
    # zero padding
    H, W, C = img.shape
    _img = np.zeros([H + 2, W + 2, C], dtype=np.float32)
    _img[1:H+1, 1:W+1, :] = img.copy()
    out = np.zeros_like(img, dtype=np.float32)

    for i in range(0, H):
        for j in range(0, H):
            for k in range(3):
                start_i = i
                end_i = i + 3
                start_j = j
                end_j = j + 3
                out[i, j, k] = np.sum(np.multiply(
                    _img[start_i:end_i, start_j:end_j, k], kernel))/16.
    return out


# Read image
img = cv2.imread("imori_noise.jpg").astype(np.float32)

# Gaussianfilter
Gaussian_kernel = [[1., 2, 1], [2., 4., 2.], [1., 2., 1.]]
out = gaussian_filter(img, Gaussian_kernel).astype(np.uint8)

# Show and save image
cv2.imwrite("Myresult/out9.jpg", out)
cv2.namedWindow("result", 0)
cv2.resizeWindow("result", 256, 256)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# =============================================================================
# OpenCV implement
# Read image
img = cv2.imread("imori_noise.jpg")

# Average pooling
out = cv2.GaussianBlur(img, (3, 3), 0)
# step size is different

# Show and save image
cv2.namedWindow("result2", 0)
cv2.resizeWindow("result2", 256, 256)
cv2.imshow("result2", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
# =============================================================================
