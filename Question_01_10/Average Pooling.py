# -*- coding: utf-8 -*-
import cv2
import numpy as np


def average_pooling(img, ksize):
    H, W, C = img.shape
    out = np.zeros_like(img, dtype=np.float32)

    for i in range(0, H, ksize):
        for j in range(0, H, ksize):
            start_i = i
            end_i = i + ksize
            start_j = j
            end_j = j + ksize
            out[start_i:end_i, start_j:end_j, :] = np.mean(
                np.mean(img[start_i:end_i, start_j:end_j, :], axis=0), axis=0)
    return out


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# Average pooling
out = average_pooling(img, 8).astype(np.uint8)

# Show and save image
cv2.imwrite("Myresult/out7.jpg", out)
cv2.namedWindow("result",0);
cv2.resizeWindow("result", 256, 256);
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# =============================================================================
# OpenCV implement
# Read image
img = cv2.imread("imori.jpg")

# Average pooling
out = cv2.blur(img, (8, 8))  # sum(square)/(8 * 8)
# step size is different

# Show and save image
cv2.namedWindow("result2",0);
cv2.resizeWindow("result2", 256, 256);
cv2.imshow("result2", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

# =============================================================================