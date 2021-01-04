# -*- coding: utf-8 -*-
import cv2
import numpy as np


# Prewitt filter
def prewitt_filter(img, K_size=3):
    H, W = img.shape

	# Zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)
    tmp = out.copy()
    
    outv = out.copy()
    outh = out.copy()

    # vertical kernel
    Kv = [[-1., -1., -1.], [0., 0., 0.], [1., 1., 1.]]
    # horizontal kernel
    Kh = [[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]]

    ## filtering
    for y in range(H):
        for x in range(W):
           outv[pad+y, pad+x] = np.sum(Kv * tmp[y:y+K_size, x:x+K_size])
           outh[pad+y, pad+x] = np.sum(Kh * tmp[y:y+K_size, x:x+K_size])    
        
    outv = np.clip(outv, 0, 255)
    outh = np.clip(outh, 0, 255)
    
    outv = outv[pad:pad+H, pad:pad+W].astype(np.uint8)
    outh = outh[pad:pad+H, pad:pad+W].astype(np.uint8)

    return outv, outh


# Read image
img = cv2.imread("imori.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Prewitt filter
outv, outh = prewitt_filter(img)

# Show and save image
out = np.hstack([outv,outh])
cv2.imwrite("Myresult/out16.jpg", out)
cv2.namedWindow("result", 0)
cv2.resizeWindow("result", 512, 256)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
