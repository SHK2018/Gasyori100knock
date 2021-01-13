# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
           outv[pad+y, pad+x] = np.sum(tmp[y:y+K_size, x:x+K_size] * Kv)
           outh[pad+y, pad+x] = np.sum(tmp[y:y+K_size, x:x+K_size] * Kh)    
        
    # outv = np.clip(outv, 0, 255)
    # outh = np.clip(outh, 0, 255)
    
    outv = outv[pad:pad+H, pad:pad+W]
    outh = outh[pad:pad+H, pad:pad+W]

    return outh, outv

# Gaussian filter
def gaussian_filter(img, K_size=3, sigma=1.3):
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape

        
    ## Zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
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
            for c in range(C):
                out[pad + y, pad + x, c] = np.sum(tmp[y: y + K_size, x: x + K_size, c] * K)

    out = out[pad: pad + H, pad: pad + W]

    return out

# Harris corner detection
def Harris_corner_step1(img):
    # Step 1: Gray scaling img
    gray = BGR2GRAY(img)
    
    # Step 2: Get Hessian matrix
    Ix, Iy = sobel_filter(gray)
    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    Ixy = Ix * Iy
    
    # Step 3: Apply Gaussian filter on Hessian matrix
    Ix2_gf = gaussian_filter(Ix2, K_size=3, sigma=3) 
    Iy2_gf = gaussian_filter(Iy2, K_size=3, sigma=3) 
    Ixy_gf = gaussian_filter(Ixy, K_size=3, sigma=3)
    
    # Print and save result
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0.2)
    
    title = ["Ix^2", "Iy^2", "Ixy"]
    result = [Ix2_gf, Iy2_gf, Ixy_gf]
    for i in range(len(title)):
        plt.subplot(1,len(title),i+1)
        plt.imshow(result[i], cmap='gray')
        plt.title(title[i])
        plt.axis("off")
    
    plt.savefig("Myresult/out82.png", dpi=326)
    plt.show()

# Read image
img = cv2.imread("Jeanne.jpg")

# Process img
Harris_corner_step1(img)
