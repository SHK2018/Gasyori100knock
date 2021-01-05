# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt

def hist_mani(img, m0=128, s0=52):
    mean = np.mean(img)
    std = np.std(img)
    out = img.copy()
    
    	# normalization
    out = s0/std * (out - mean) + m0
    out = np.clip(out, 0, 255).astype(np.uint8)
    
    return out


img = cv2.imread("imori_dark.jpg")
out = hist_mani(img)

# Display histogram
plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.show()
cv2.namedWindow("result", 0)
cv2.resizeWindow("result", 256, 256)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("Myresult/out22.jpg", out)
cv2.destroyAllWindows()