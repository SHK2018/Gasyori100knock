# -*- coding: utf-8 -*-
import cv2
import numpy as np


# DCT hyoer-parameter
T = 8
K = 4

# DCT weight
def w(x, y, u, v):
    cu = 1.
    cv = 1.
    if u == 0:
        cu /= np.sqrt(2.)
    if v == 0:
        cv /= np.sqrt(2.)
    return (2*cu*cv/T) * (np.cos((2*x+1)*u*np.pi/2/T)) * (np.cos((2*y+1)*v*np.pi/2/T))


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

# MSE
def MSE(img, out):
    mse = np.sum((img - out) ** 2) / (H * W * channel)
    return mse

# PSNR
def PSNR(mse, vmax=255):
    return 10 * np.log10(vmax * vmax / mse)

# bitrate
def BITRATE():
    return 1. * T * K * K / T / T

# BGR2YCbCr
def BGR2YCbCr(img):
    out = np.zeros([H, W, channel], dtype=np.float32)
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]
    
    Y = 0.2990 * R + 0.5870 * G + 0.1140 * B
    Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
    Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128
    out[:, :, 0] = Y
    out[:, :, 1] = Cb
    out[:, :, 2] = Cr

    return out

# YCbCr2BGR
def YCbCr2BGR(img_):
    out = np.zeros([H, W, channel], dtype=np.float32)
    out[..., 2] = img_[..., 0] + (img_[..., 2] - 128.) * 1.4020
    out[..., 1] = img_[..., 0] - (img_[..., 1] - 128.) * 0.3441 - (img_[..., 2] - 128.) * 0.7139
    out[..., 0] = img_[..., 0] + (img_[..., 1] - 128.) * 1.7718
    
    # clipping
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out

def quantization(F):
    
    Q1 = np.array(((16, 11, 10, 16, 24, 40, 51, 61),
                   (12, 12, 14, 19, 26, 58, 60, 55),
                   (14, 13, 16, 24, 40, 57, 69, 56),
                   (14, 17, 22, 29, 51, 87, 80, 62),
                   (18, 22, 37, 56, 68, 109, 103, 77),
                   (24, 35, 55, 64, 81, 104, 113, 92),
                   (49, 64, 78, 87, 103, 121, 120, 101),
                   (72, 92, 95, 98, 112, 100, 103, 99)), dtype=np.float32)
    
    Q2 = np.array(((17, 18, 24, 47, 99, 99, 99, 99),
                   (18, 21, 26, 66, 99, 99, 99, 99),
                   (24, 26, 56, 99, 99, 99, 99, 99),
                   (47, 66, 99, 99, 99, 99, 99, 99),
                   (99, 99, 99, 99, 99, 99, 99, 99),
                   (99, 99, 99, 99, 99, 99, 99, 99),
                   (99, 99, 99, 99, 99, 99, 99, 99),
                   (99, 99, 99, 99, 99, 99, 99, 99)), dtype=np.float32)
    
    Q = Q1
    for c in range(channel):
        if c > 0:
            Q = Q2
        for ys in range(0, H, T):
            for xs in range(0, W, T):
                F[ys: ys + T, xs: xs + T,c] = np.round(F[ys: ys + T, xs: xs + T, c] / Q) * Q
                
    return F


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)
H, W, channel = img.shape

# BGR2YCbCr
img_ = BGR2YCbCr(img)

# DCT
F = dct(img_)

# quantization
F_ = quantization(F)

# IDCT
out_ = idct(F_)

# YCbCr2BGR
out = YCbCr2BGR(out_)

# MSE
mse = MSE(img, out)

# PSNR
psnr = PSNR(mse)

# bitrate
bitrate = BITRATE()

print("MSE:", mse)
print("PSNR:", psnr)
print("bitrate:", bitrate)

# Save result
cv2.namedWindow("result", 0)
cv2.resizeWindow("result", 256, 256)

cv2.imshow("result", out)
cv2.waitKey(0)

cv2.imwrite("Myresult/out40.jpg", out)

cv2.destroyAllWindows()
