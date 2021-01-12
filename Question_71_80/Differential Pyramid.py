# -*- coding: utf-8 -*-
import cv2
import numpy as np


def BGR2GRAY(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    out = 0.2126*r + 0.7152*g + 0.0722*b
    out = out.astype(np.uint8)

    return out

def bilinear_interpolation(img, ax=1, ay=1):
    
    if len(img.shape) > 2:
        H, W, C = img.shape
    else:
        H, W = img.shape
        C = 1

    aH = int(ay * H)
    aW = int(ax * W)

    # get position of resized image
    y = np.arange(aH).repeat(aW).reshape(aW, -1)
    x = np.tile(np.arange(aW), (aH, 1))

    # get position of original position
    y = (y / ay)
    x = (x / ax)

    ix = np.floor(x).astype(np.int)
    iy = np.floor(y).astype(np.int)

    ix = np.minimum(ix, W-2)
    iy = np.minimum(iy, H-2)

    # get distance 
    dx = x - ix
    dy = y - iy

    if C > 1:
        dx = np.repeat(np.expand_dims(dx, axis=-1), C, axis=-1)
        dy = np.repeat(np.expand_dims(dy, axis=-1), C, axis=-1)

    # interpolation
    out = (1-dx) * (1-dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix+1] + (1 - dx) * dy * img[iy+1, ix] + dx * dy * img[iy+1, ix+1]

    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out


# Read image
img = cv2.imread("Jeanne.jpg").astype(np.float)

# Gray scaling
gray = BGR2GRAY(img)

# Process image
_gray = bilinear_interpolation(gray, 0.5, 0.5)
out = bilinear_interpolation(_gray, 2, 2)

out = np.abs(out - gray)
out = out / out.max() * 255
out = out.astype(np.uint8)

# Show and save result
# cv2.namedWindow("result", 1)
# cv2.resizeWindow("result", 256, 256)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("Myresult/out74.jpg", out)
cv2.destroyAllWindows()
