# -*- coding: utf-8 -*-
import cv2
import numpy as np


def rotation(img, a=1, b=0, c=0, d=1, tx=0, ty=0):
    H, W, C = img.shape
    
    # new H and w for output image
    H_new = max(int(H*d), H)
    W_new = max(int(W*a), W)
    out = np.zeros((H_new, W_new, C))
    
    # kernel
    K = np.matrix([[d, -b],[-c, a]])
    adbc = a*d - b*c
    
    K_temp = np.matrix([[a, b],[c, d]])
    pixel_temp = (K_temp * np.matrix([[int(H/2.), int(W/2.)]]).T).astype(np.int)
    T = np.matrix([int(H*d/2), int(W*a/2)]).T - pixel_temp
    
    for j in range(H_new):
        for i in range(W_new):
            pixel = (np.matmul(K, np.matrix([i, j]).T)/adbc - T).astype(np.int)
            if (pixel[1,0]<H) & (np.min(pixel)>=0) & (pixel[0,0]<W):
                out[j, i, :] = img[pixel[1, 0], pixel[0, 0], :]
    return out.astype(np.uint8)


# Read image
img = cv2.imread("imori.jpg").astype(np.float)

# Affine
A = 30.
theta = - np.pi * A / 180.

# Process image
out = rotation(img, a=np.cos(theta), b=-np.sin(theta), c=np.sin(theta), d=np.cos(theta),
               tx=0, ty=0)

# Show and save result
cv2.namedWindow("result", 0)
cv2.resizeWindow("result", 256, 256)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("Myresult/out30.jpg", out)
cv2.destroyAllWindows()
