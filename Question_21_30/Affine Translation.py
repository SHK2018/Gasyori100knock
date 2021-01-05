# -*- coding: utf-8 -*-
import cv2
import numpy as np


def translation(img, tx=0, ty=0):
    H, W, C = img.shape
    out = np.zeros_like(img)
    
    # kernel
    K = np.matrix([[1, 0, tx],[0, 1, ty],[0, 0, 1]])
    
    for j in range(H):
        for i in range(W):
            pixel = np.matmul(K, np.matrix([i, j, 1]).T)
            if ((np.max(pixel)<H) & (np.min(pixel)>=0)):
                out[pixel[1, 0], pixel[0, 0], :] = img[j, i, :]
    return out.astype(np.uint8)

# Read image
img = cv2.imread("imori.jpg").astype(np.float)

# Process image
out = translation(img, tx=30, ty=-30)

# Show and save result
cv2.namedWindow("result", 0)
cv2.resizeWindow("result", 256, 256)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("Myresult/out28.jpg", out)
cv2.destroyAllWindows()