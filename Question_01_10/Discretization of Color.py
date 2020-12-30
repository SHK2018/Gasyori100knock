# -*- coding: utf-8 -*-
import cv2
import numpy as np

def decrease_color(img):
    out = img.copy()
    
    out = out // 64 * 64 + 32
    
    return out

# Read image
img = cv2.imread("imori.jpg")

# Dcrease color
out = decrease_color(img)

cv2.imwrite('Myresult/out6.jpg', out)
cv2.namedWindow("result",0);
cv2.resizeWindow("result", 256, 256);
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
