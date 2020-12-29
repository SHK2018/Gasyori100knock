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


# Read image
img = cv2.imread("./imori.jpg").astype(float)

# Grayscale
out = BGR2GRAY(img)

# Save result
cv2.imwrite("Myresult/out2.jpg", out)
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
cv2.imshow("result2", out2)
cv2.waitKey(0)
cv2.destroyAllWindows()
# =============================================================================
