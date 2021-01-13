# -*- coding: utf-8 -*-
import cv2
import numpy as np


## Grayscale
def BGR2GRAY(img):
    gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
    gray = gray.astype(np.uint8)
    return gray

# Sobel filter
def sobel_filter(img, K_size=3):
    H, W = img.shape

    # Zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float32)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float32)
    tmp = out.copy()
    
    outv = out.copy()
    outh = out.copy()

    # vertical kernel
    Kv = [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]
    # horizontal kernel
    Kh = [[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]

    ## filtering
    for y in range(H):
        for x in range(W):
           outv[pad+y, pad+x] = np.mean(tmp[y:y+K_size, x:x+K_size] * Kv)
           outh[pad+y, pad+x] = np.mean(tmp[y:y+K_size, x:x+K_size] * Kh)    
        
    # outv = np.clip(outv, 0, 255)
    # outh = np.clip(outh, 0, 255)
    
    outv = outv[pad:pad+H, pad:pad+W]
    outh = outh[pad:pad+H, pad:pad+W]

    return outh, outv

# Gaussian filter
def gaussian_filter(img, K_size=3, sigma=1.3):
    H, W = img.shape

    ## Zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)

    ## prepare Kernel
    K = np.zeros((K_size, K_size), dtype=np.float)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (2 * np.pi * sigma * sigma)
    K /= K.sum()

    tmp = out.copy()

    # filtering
    for y in range(H):
        for x in range(W):
                out[pad + y, pad + x] = np.sum(tmp[y: y + K_size, x: x + K_size] * K)

    out = out[pad: pad + H, pad: pad + W]

    return out

# corner detect
def corner_detect(gray, Ix2, Iy2, Ixy, k=0.04, th=0.1):
    H, W = gray.shape

    # prepare for show detection
    out = np.array((gray, gray, gray))
    out = np.transpose(out, (1, 2, 0))

    # get Hessian value
    Hes = np.zeros((H, W))

    for y in range(H):
        for x in range(W):
            Hes[y, x] = Ix2[y, x] * Iy2[y, x] - Ixy[y, x] ** 2

    ## Detect Corner and show
    for y in range(H):
        for x in range(W):
            if Hes[y, x] == np.max(Hes[max(y-1, 0): min(y+2, H), max(x-1, 0): min(x+2, W)]) and Hes[y, x] > np.max(Hes) * 0.1:
                out[y, x] = [0, 0, 255]

    out = out.astype(np.uint8)

    return out

# Harris corner detection
def Harris_corner(img):
    # Step 1: Gray scaling img
    gray = BGR2GRAY(img)
    
    # Step 2: Get Hessian matrix
    Ix, Iy = sobel_filter(gray)
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy
    
    # Step 3: Corner detect
    out = corner_detect(gray, Ix2, Iy2, Ixy)

    return out


# Read image
img = cv2.imread("Jeanne.jpg")

# Harris corner detection
out = Harris_corner(img)

cv2.imwrite("Myresult/out81.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
