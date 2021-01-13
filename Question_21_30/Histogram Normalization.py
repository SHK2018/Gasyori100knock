# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt

def grayscale_transformation(img, a=0, b=255):
    c = np.min(img)
    d = np.max(img)
    out = img.copy()
    
    # normalization
    out = (b-a) / (d - c) * (out - c) + a
    out[out < a] = a
    out[out > b] = b
    out = out.astype(np.uint8)
    
    return out


img = cv2.imread("imori_dark.jpg")
out = grayscale_transformation(img)

# Display histogram
plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.show()
cv2.namedWindow("result", 0)
cv2.resizeWindow("result", 256, 256)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("Myresult/out21.jpg", out)
cv2.destroyAllWindows()