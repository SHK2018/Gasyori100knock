# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt

def hist_equal(img, z_max=255):
	H, W, C = img.shape
	S = H * W * C * 1.

	out = img.copy()

	sum_h = 0.

	for i in range(1, 255):
		ind = np.where(img == i)
		sum_h += len(img[ind])
		z_prime = z_max / S * sum_h
		out[ind] = z_prime

	out = out.astype(np.uint8)

	return out


img = cv2.imread("imori.jpg")
out = hist_equal(img)

# Display histogram
plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.show()
cv2.namedWindow("result", 0)
cv2.resizeWindow("result", 256, 256)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("Myresult/out23.jpg", out)
cv2.destroyAllWindows()