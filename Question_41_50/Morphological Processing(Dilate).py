# -*- coding: utf-8 -*-
import numpy as np
import cv2

# Gray scale
def BGR2GRAY(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    # Gray scale
    out = 0.2126 * r + 0.7152 * g + 0.0722 * b
    out = out.astype(np.uint8)

    return out

# Otsu Binalization
def otsu_binarization(img):
    max_sigma = 0
    max_t = 0
    H, W = img.shape
    out = img.copy()
    
    # determine threshold
    for _t in range(1, 256):
        v0 = out[np.where(out < _t)]
        m0 = np.mean(v0) if len(v0) > 0 else 0.
        w0 = len(v0) / (H * W)
        v1 = out[np.where(out >= _t)]
        m1 = np.mean(v1) if len(v1) > 0 else 0.
        w1 = len(v1) / (H * W)
        sigma = w0 * w1 * ((m0 - m1) ** 2)
        if sigma > max_sigma:
            max_sigma = sigma
            max_t = _t

    # Binarization
    print("threshold >>", max_t)
    th = max_t
    out[out < th] = 0
    out[out >= th] = 255

    return out

def dilate(img, Dil_time=1):
    out = img.copy()
    H, W = img.shape
    
    # Output image, zero padding
    out = np.zeros([H+2, W+2], dtype = np.int)
    
    # Kernel
    K = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    
    # Apply kernel on the image
    for z in range(Dil_time):
        out[1:H+1, 1:W+1] = img.copy()
        for j in range(1, H+1):
            for i in range(1, W+1):
                if out[j, i] == 0:
                    if np.sum(K * out[j-1:j+2, i-1:i+2]) >= 255:
                        img[j-1, i-1] = 255
    
    return img

# Read image
img = cv2.imread("imori.jpg")

# Gray Scale
out1 = BGR2GRAY(img)

# Otsu Binarization
out2 = otsu_binarization(out1)

# Dilate
out = dilate(out2, 2)

# Show and save image
cv2.namedWindow("result")
cv2.resizeWindow("result", 256, 256)

cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("Myresult/out47.jpg", out)
