# -*- coding: utf-8 -*-
import cv2
import numpy as np


def BGR2GRAY(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    out = 0.2126*r + 0.7152*g + 0.0722*b
    out = out.astype(np.uint8)

    return out

def L4N(img):
        
    H, W = img.shape
    
    # source = np.zeros_like(img, dtype=np.uint8)
    # destination = np.zeros_like(img, dtype=np.uint8)
    out = np.zeros([H + 1, W + 1], dtype=np.uint8)

    
    K = np.array([[0, 1],[1, 0]], dtype=np.uint8)
    tag = 1
    
    for j in range(1, H+1):
        for i in range(1, W+1):
            temp = K * out[j-1:j+1, i-1:i+1]
            if img[j-1, i-1]:
                if np.sum(temp) > 0:
                    #source[j-1, i-1] = tag
                    out[j, i] = np.min(temp[temp>0])
                    continue
                out[j, i] = tag
                tag += 1
 
    return (out[1:H+1, 1:W+1]*10).astype(np.uint8)
# Read image
img = cv2.imread("seg.png").astype(np.float32)

# Gray scale
gray = BGR2GRAY(img)

# Alpha blending
out = L4N(gray)

# Show and save image
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("Mysult/out59.jpg", out)
