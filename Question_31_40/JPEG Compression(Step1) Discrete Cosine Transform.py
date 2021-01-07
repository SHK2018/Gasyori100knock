# -*- coding: utf-8 -*-
import cv2
import numpy as np


# DCT hyoer-parameter
T = 8
K = 8

# DCT weight
def w(x, y, u, v):
    cu = 1.
    cv = 1.
    if u == 0:
        cu /= np.sqrt(2.)
    if v == 0:
        cv /= np.sqrt(2.)
    return (2*cu*cv/T) * (np.cos((2*x+1)*u*np.pi/2./T)) * (np.cos((2*y+1)*v*np.pi/2./T))


# DCT
def dct(img):
    F = np.zeros((H, W, channel), dtype=np.float32)
    for c in range(channel):
        for yi in range(0, H, T):
            for xi in range(0, W, T):   # crop the image into 8 X 8
                for v in range(0, T):
                    for u in range(0, T):   # each(u, v) in K space represent a cosine wave
                        for y in range(0, T):
                            for x in range(0, T):   # each cosine wave contribute to the entire image
                                F[v+yi, u+xi, c] += img[y+yi, x+xi, c] * w(x,y,u,v)
    return F

# IDCT
def idct(F):
    out = np.zeros((H, W, channel), dtype=np.float32)
    for c in range(channel):
        for yi in range(0, H, T):
            for xi in range(0, W, T):   # crop the image into 8 X 8
                for y in range(0, T):
                    for x in range(0, T):   # each(u, v) in K space represent a cosine wave
                        for v in range(0, K):
                            for u in range(0, K):   # each cosine wave contribute to the entire image
                                out[y+yi, x+xi, c] += F[v+yi, u+xi, c] * w(x,y,u,v)
    
    # clipping
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)
H, W, channel = img.shape

# DCT
F = dct(img)

# IDCT
out = idct(F)

# Save result
cv2.namedWindow("result", 0)
cv2.resizeWindow("result", 256, 256)

cv2.imshow("result", out)
cv2.waitKey(0)

cv2.imwrite("Myresult/out36.jpg", out)

cv2.destroyAllWindows()
