# -*- coding: utf-8 -*-
import numpy as np
import cv2


def connected_8(img):
    H, W, C = img.shape
    
    out = np.zeros((H, W, C), dtype=np.uint8)
    
    _temp = np.zeros([H, W], dtype=np.int)
    _temp[img[:, :, 0] > 0] = 1
    
    _temp = 1- _temp
    
    for y in range(H):
        for x in range(W):
            if _temp[y, x] > 0:
                continue
            S = 0
            S += _temp[y, min(x+1, W-1)] - _temp[y, min(x+1, W-1)] * _temp[max\
                (y-1, 0), min(x+1, W-1)] * _temp[max(y-1, 0), x]
            S += _temp[max(y-1, 0), x] - _temp[max(y-1, 0), x] * _temp[max\
                (y-1, 0), max(x-1, 0)] * _temp[y, max(x-1, 0)]
            S += _temp[y, max(x-1, 0)] - _temp[y, max(x-1, 0)] * _temp[min\
                (y+1, H-1), max(x-1, 0)] * _temp[min(y+1, H-1), x]
            S += _temp[min(y+1, H-1), x] - _temp[min(y+1, H-1), x] * _temp[min\
                (y+1, H-1), min(x+1, W-1)] * _temp[y, min(x+1, W-1)]
    
            if S == 0:
                out[y,x] = [0, 0, 255]
            elif S == 1:
                out[y,x] = [0, 255, 0]
            elif S == 2:
                out[y,x] = [255, 0, 0]
            elif S == 3:
                out[y,x] = [255, 255, 0]
            elif S == 4:
                out[y,x] = [255, 0, 255]
                    
    out = out.astype(np.uint8)

    return out


# Read image
img = cv2.imread("renketsu.png")

#

# Process image
out = connected_8(img)

# Show and save image
cv2.namedWindow("result", 0)
cv2.resizeWindow("result", 300, 300)

cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("Myresult/62.jpg", out)

