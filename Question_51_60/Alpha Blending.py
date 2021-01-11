# -*- coding: utf-8 -*-
import cv2
import numpy as np


def alpha_blending(img1, img2, alpha=0.5):
    out = img1 * alpha + img2 * (1-alpha)
 
    return out.astype(np.uint8)

# Read image
img1 = cv2.imread("imori.jpg").astype(np.float32)
img2 = cv2.imread("thorino.jpg").astype(np.float32)

# Alpha blending
out = alpha_blending(img1, img2, 0.5)

# Show and save image
cv2.namedWindow("result", 0)
cv2.imshow("result", out)
cv2.waitKey(0)

cv2.imwrite("Myresult/out60.jpg", out)
cv2.destroyAllWindows()