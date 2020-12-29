# -*- coding: utf-8 -*-
import cv2
import numpy as np


def mean_gray(img, num, th=1):
    # get pixel that whose value greater than threshold
    num0 = np.sum(img >= th)  # pixel number that greater thant threshold
    num1 = num - num0
    mean0 = sum(img[img >= th])/num0 if num0 > 0 else 0
    mean1 = sum(img[img < th])/num1 if num0 > 1 else 0
    return num0/num, mean0, num1/num, mean1


def binarizationOfOtsu(img):
    H, W = img.shape      # get image height and width
    num = H * W     # total numer of pixel
    sdv = 0
    threshold = 0
    for t in range(255):
        num0, mean0, num1, mean1 = mean_gray(img, num, t)
        g = num0*num1*((mean0-mean1)**2)
        if g > sdv:
            sdv = g
            threshold = t

    return threshold


def BGR2GRAY(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    out = 0.2126*r + 0.7152*g + 0.0722*b
    out = out.astype(np.uint8)

    return out


def binarization(img, threshold=128):
    # [img >= threshold] return a turple that contains all the index of value meet the requirement.
    img[img >= threshold] = 255
    img[img < threshold] = 0
    return img


# Read image
img = cv2.imread("Myresult/imori.jpg").astype(np.float32)

# Grayscale
out = BGR2GRAY(img)

# Binarization of Otsu
th = binarizationOfOtsu(out)
print("The proper threshold is %03d" % th)
out = binarization(out, th)

# Save result
cv2.imwrite("./out4.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# =============================================================================
# OpenCV implement:
import cv2
import numpy as np
img = cv2.imread("imori.jpg")
# In OpenCV API data type will change by default when doing grayscale
out2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
r, out2 = cv2.threshold(out2, 1, 256, cv2.THRESH_OTSU)
print("The proper threshold is %03d" % r)
cv2.imshow("result2", out2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# =============================================================================
