# -*- coding: utf-8 -*-
import cv2
import numpy as np


# Emboss filter
def LoG_filter(img, K_size=5, sigma=3):
    H, W = img.shape

	# Zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)
    tmp = out.copy()

    # LoG Kernel
    K = np.zeros((K_size, K_size), dtype=np.float)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = (x ** 2 + y ** 2 - sigma ** 2) * np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    
    K /= (2 * np.pi * (sigma ** 6))
    # Normalization
    K /= K.sum()

    ## filtering
    for y in range(H):
        # for x in range(W):
           out[pad+y, pad+x] = np.sum(K * tmp[y:y+K_size, x:x+K_size]) 
        
    out = np.clip(out, 0, 255)

    out = out[pad:pad+H, pad:pad+W].astype(np.uint8)

    return out


# Read image
img = cv2.imread("imori_noise.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Emboss filter
out = LoG_filter(img, 5 ,3)

# Show and save image
cv2.imwrite("Myresult/out19.jpg", out)
cv2.namedWindow("result", 0)
cv2.resizeWindow("result", 256, 256)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
