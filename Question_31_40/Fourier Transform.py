# -*- coding: utf-8 -*-
import cv2
import numpy as np

def dft(img):
    H, W, C = img.shape
    out = np.zeros_like(img).astype(np.complex)
    for k in range(H):
        for l in range(W):
            temp = 0.;
            for m in range(H):
                for n in range(W):
                    temp_exp = -2j * np.pi * (k * n / W + l * m / H)
                    temp += img[m, n] * np.exp(temp_exp) / np.sqrt(H * W)
            out[k, l] = temp;
                    
    return out


img = cv2.imread("imori_gray.jpg")

out = dft(img)

cv2.namedWindow("result")
cv2.resizeWindow("result", 256, 256)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("Myresult/out32.jpg", out)
cv2.destroyAllWindows()

