# -*- coding: utf-8 -*-
import cv2


def BGR2RGB(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    #BRG2RGB
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b

    return img

# Read image


img = cv2.imread("./imori.jpg")

# Channel swapping
img = BGR2RGB(img)

#Save result
cv2.imwrite("Myresult/out1.jpg", img)
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# =============================================================================
# OpenCV2 implement:
import cv2
import numpy as np
img = cv2.imread("imori.jpg")
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("result2", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
# =============================================================================
