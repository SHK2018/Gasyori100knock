# -*- coding: utf-8 -*-
import numpy as np
import cv2


def SSD(img, part):
    out = img.copy()
    
    H, W, C = img.shape
    h, w, c = part.shape
    
    S = 255**2 * h * w * c
    y, x = -1, -1    
    for j in range(H - h):
        for i in range(W - w):
            temp_S = np.sum((out[j:j+h, i:i+w] - part)**2)
            if temp_S < S:
                S = temp_S
                y, x = j, i
    
    # draw rectangle
    cv2.rectangle(out, pt1=(x, y), pt2=(x+w, y+h), color=(0,0,255), thickness=1)
    out = out.astype(np.uint8)
    
    return out
    
# Read image
img = cv2.imread("imori.jpg").astype(np.float32)
part = cv2.imread("imori_part.jpg").astype(np.float32)

# Process image
out = SSD(img, part)

# Show and save image
cv2.namedWindow("result", 0)
cv2.resizeWindow("result", 256, 256)

cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("Myresult/out54.jpg", out)

