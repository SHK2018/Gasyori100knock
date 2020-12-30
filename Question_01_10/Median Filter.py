# -*- coding: utf-8 -*-
import cv2
import numpy as np


def median_filter(img, ksize):
    # zero padding
    H, W, C = img.shape
    _img = np.zeros([H + 2, W + 2, C], dtype=np.float32)
    _img[1:H+1, 1:W+1, :] = img.copy()
    out = np.zeros_like(img, dtype=np.float32)

    for i in range(0, H):
        for j in range(0, H):
            for k in range(3):
                start_i = i
                end_i = i + ksize
                start_j = j
                end_j = j + ksize
                out[i, j, k] = np.median(_img[start_i:end_i, start_j:end_j, k])
    return out


# Read image
img = cv2.imread("imori_noise.jpg").astype(np.float32)

# Gaussianfilter
out = median_filter(img, 3).astype(np.uint8)

# Show and save image
cv2.imwrite("Myresult/out10.jpg", out)
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
out = cv2.medianBlur(img, 3)
# step size is different

# Show and save image
cv2.namedWindow("result2", 0)
cv2.resizeWindow("result2", 256, 256)
cv2.imshow("result2", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
# =============================================================================
