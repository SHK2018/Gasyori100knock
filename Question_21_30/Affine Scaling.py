# -*- coding: utf-8 -*-
import cv2
import numpy as np


def scaling_translation(img, a = 1, d = 1, tx=0, ty=0):
    H, W, C = img.shape
    
    # new H and w for output image
    H_new = int(H*d)
    W_new = int(W*a)
    out = np.zeros((H_new, W_new, C))
    
    # kernel
    K = np.matrix([[d, 0],[0, a]])
    T = np.matrix([tx, ty]).T
    adbc = a*d - 0*0
    
    for j in range(H_new):
        for i in range(W_new):
            pixel = (np.matmul(K, np.matrix([i, j]).T)/adbc - T).astype(np.int)
            if (pixel[1,0]<H) & (np.min(pixel)>=0) & (pixel[0,0]<W):
                out[j, i, :] = img[pixel[1, 0], pixel[0, 0], :]
    return out.astype(np.uint8)


# Read image
img = cv2.imread("imori.jpg").astype(np.float)

# Process image
out = scaling_translation(img, a=1.3, d=0.8, tx=30, ty=-30)

# Show and save result
cv2.namedWindow("result", 0)
cv2.resizeWindow("result", 256, 256)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("Myresult/out29.jpg", out)
cv2.destroyAllWindows()