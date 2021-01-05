# -*- coding: utf-8 -*-
import cv2
import numpy as np

def NN_interpolation(img, ax = 1, ay = 1):
    # input and out size
    H, W, C = img.shape
    aH = int(H * ay)
    aW = int(W * ax)
    
    out = np.zeros((aW, aH, C), dtype = np.uint8)
    
    for i in range(aW):
        for j in range(aH):
            out[j, i, :] = img[round(j/ay), round(i/ax), :]
    
    return out


# Read image
img = cv2.imread("imori.jpg")

# Process image
out = NN_interpolation(img, 1.5, 1.5)

# Show and save result
cv2.namedWindow("result", 0)
cv2.resizeWindow("result", 256, 256)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("Myresult/out25.jpg", out)
cv2.destroyAllWindows()