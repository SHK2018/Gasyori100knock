# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt


def gamma_correction(img, c=1, g=2.2):
	out = img.copy()
	out = out.astype(np.float32)/255.
	out = (1/c) * (out) ** (1/g)

	out *= 255
	out = out.astype(np.uint8)

	return out


img = cv2.imread("imori_gamma.jpg")
out = gamma_correction(img)

# Display histogram
plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.show()
cv2.namedWindow("result", 0)
cv2.resizeWindow("result", 256, 256)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("Myresult/out24.jpg", out)
cv2.destroyAllWindows()
