# -*- coding: utf-8 -*-
import cv2
import numpy as np


def Skew(img, a=1, b=0, c=0, d=1, tx=0, ty=0):
    H, W, C = img.shape
    
    # prepare affine image temporary
    H_new = np.ceil(dy + H).astype(np.int)
    W_new = np.ceil(dx + W).astype(np.int)
    out = np.zeros((H_new, W_new, C), dtype=np.float32)
    # kernel
    K = np.matrix([[d, -b],[-c, a]])
    
    adbc = a*d - b*c
    max_x = 0
    max_y = 0
    min_x = 2*W
    min_y = 2*H
           
    for j in range(H_new):
        for i in range(W_new):
            pixel = (np.matmul(K, np.matrix([i, j]).T)/adbc).astype(np.int)
            if (pixel[1,0]<H) & (np.min(pixel)>=0) & (pixel[0,0]<W):
                out[j, i, :] = img[pixel[1, 0], pixel[0, 0], :]
    return out.astype(np.uint8)



# Read image
img = cv2.imread("imori.jpg").astype(np.float)

# Affine
dx = 30.
dy = 30.
H, W, C = img.shape

# Process image
out = Skew(img, a=1, b=dx/H, c=dy/W, d=1, tx=0, ty=0)

# Show and save result
cv2.namedWindow("result", 0)
cv2.resizeWindow("result", 256, 256)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("Myresult/out31.jpg", out)
cv2.destroyAllWindows()
