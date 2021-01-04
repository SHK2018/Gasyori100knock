# -*- coding: utf-8 -*-
import cv2
import numpy as np


# Median filter
def median_filter(img, K_size=3):
    H, W, C = img.shape

    ## Zero padding
    pad = K_size // 2
    out = np.zeros((H + pad*2, W + pad*2, C), dtype=np.float)
    out[pad:pad+H, pad:pad+W] = img.copy().astype(np.float)

    tmp = out.copy()

    # filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad+y, pad+x, c] = np.median(tmp[y:y+K_size, x:x+K_size, c])

    out = out[pad:pad+H, pad:pad+W].astype(np.uint8)

    return out


# Read image
img = cv2.imread("imori_noise.jpg")

# Median filter
out = median_filter(img)

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
