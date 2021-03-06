# -*- coding: utf-8 -*-
import cv2
import numpy as np


# Motion filter
def motion_filter(img, K_size=3):
    H, W, C = img.shape

    ## Zero padding
    pad = K_size // 2
    out = np.zeros((H + pad*2, W + pad*2, C), dtype=np.float)
    out[pad:pad+H, pad:pad+W] = img.copy().astype(np.float)

    tmp = out.copy()
    
    ## prepare Kernel
    K = np.zeros((K_size, K_size), dtype=np.float)
    K = np.eye(K_size, dtype = np.float32)/K_size

    ## filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad+y, pad+x, c] = np.sum(tmp[y:y+K_size, x:x+K_size, c] * K)

    out = out[pad:pad+H, pad:pad+W].astype(np.uint8)

    return out


# Read image
img = cv2.imread("imori.jpg")

# motion filter
out = motion_filter(img)

# Show and save image
cv2.imwrite("Myresult/out12.jpg", out)
cv2.namedWindow("result", 0)
cv2.resizeWindow("result", 256, 256)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()