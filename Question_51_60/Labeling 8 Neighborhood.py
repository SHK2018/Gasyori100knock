# -*- coding: utf-8 -*-
import cv2
import numpy as np


def L8N(img):
    out = img.copy()
 
    return out.astype(np.uint8)

# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# Alpha blending
out = L8N(img)

# Show and save image
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("Mysult/out59.jpg", out)
