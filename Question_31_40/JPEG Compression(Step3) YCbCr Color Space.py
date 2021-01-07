# -*- coding: utf-8 -*-
import cv2
import numpy as np


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

# MSE
def MSE(img, out):
    mse = np.sum((img - out) ** 2) / (H * W * channel)
    return mse

# PSNR
def PSNR(mse, vmax=255):
    return 10 * np.log10(vmax * vmax / mse)


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)
H, W, channel = img.shape

# BGR2YCbCr
out = BGR2YCbCr(img)

# process image
out[..., 0] *= 0.7

# YCbCr2BGR
out = YCbCr2BGR(out)

# MSE
mse = MSE(img, out)

# PSNR
psnr = PSNR(mse)

print("MSE:", mse)
print("PSNR:", psnr)

# Save result
cv2.namedWindow("result", 0)
cv2.resizeWindow("result", 256, 256)

cv2.imshow("result", out)
cv2.waitKey(0)

cv2.imwrite("Myresult/out39.jpg", out)

cv2.destroyAllWindows()
