# -*- coding: utf-8 -*-
import cv2
import numpy as np


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
				out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])

	out = np.clip(out, 0, 255)
	out = out[pad: pad + H, pad: pad + W].astype(np.uint8)

	return out


# Read image
img = cv2.imread("imori_noise.jpg").astype(np.float32)

# Gaussianfilter
out = gaussian_filter(img).astype(np.uint8)

# Show and save image
cv2.imwrite("Myresult/out9.jpg", out)
cv2.namedWindow("result", 0)
cv2.resizeWindow("result", 256, 256)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# =============================================================================
# OpenCV implement
# Read image
img = cv2.imread("imori_noise.jpg")

# Average pooling
out = cv2.GaussianBlur(img, (3, 3), 0)
# step size is different

# Show and save image
cv2.namedWindow("result2", 0)
cv2.resizeWindow("result2", 256, 256)
cv2.imshow("result2", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
# =============================================================================
