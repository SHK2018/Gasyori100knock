# -*- coding: utf-8 -*-
import cv2
import numpy as np


# MaxMin filter
def MaxMin_filter(img, K_size=3):
    H, W = img.shape

    ## Zero padding
    pad = K_size // 2
    out = np.zeros((H + pad*2, W + pad*2), dtype=np.float)
    out[pad:pad+H, pad:pad+W] = img.copy().astype(np.float)

    tmp = out.copy()

    ## filtering
    for y in range(H):
        for x in range(W):
            out[pad+y, pad+x] = np.max(tmp[y:y+K_size, x:x+K_size]) - \
                np.min(tmp[y:y+K_size, x:x+K_size])

    out = out[pad:pad+H, pad:pad+W].astype(np.uint8)

    return out


# Read image
img = cv2.imread("imori.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# MaxMin filter
out = MaxMin_filter(img)

# Show and save image
cv2.imwrite("Myresult/out13.jpg", out)
cv2.namedWindow("result", 0)
cv2.resizeWindow("result", 256, 256)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
