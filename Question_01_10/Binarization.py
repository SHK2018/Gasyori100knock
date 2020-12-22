# -*- coding: utf-8 -*-
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


def binarization(img, threshold=128):
    # [img >= threshold] return a turple that contains all the index of value meet the requirement.
    img[img >= threshold] = 255
    img[img < threshold] = 0
    return img


# Read image
img = cv2.imread("./imori.jpg").astype(float)

# Grayscale
out = BGR2GRAY(img)

# Binarization
out = binarization(out)

# Save result
cv2.imwrite("./out3.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()


# =============================================================================
# OpenCV implement:
img = cv2.imread("Myresult/imori.jpg")
# In OpenCV API data type will change by default when doing grayscale
out2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
r, out2 = cv2.threshold(out2, 127, 255, cv2.THRESH_OTSU)
cv2.imshow("result2", out2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# =============================================================================
