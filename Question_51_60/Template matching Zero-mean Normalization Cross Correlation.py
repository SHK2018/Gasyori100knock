# -*- coding: utf-8 -*-
import numpy as np
import cv2


def ZNCC(img, part):
    out = img.copy()

    H, W, _ = img.shape
    h, w, _ = part.shape
    
    _img = img - np.mean(img, axis=(0, 1))
    _part = part - np.mean(part, axis=(0, 1))

    S = 0
    y, x = -1, -1
    for j in range(H - h):
        for i in range(W - w):
            temp_S = np.sum(np.abs(_img[j:j+h, i:i+w] * _part)) / (
                np.sqrt(np.sum(_img[j:j+h, i:i+w] ** 2)) * np.sqrt(np.sum(_part ** 2)))
            if temp_S > S:
                S = temp_S
                y, x = j, i

    # draw rectangle
    cv2.rectangle(out, pt1=(x, y), pt2=(x+w, y+h),
                  color=(0, 0, 255), thickness=1)
    out = out.astype(np.uint8)

    return out


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)
part = cv2.imread("imori_part.jpg").astype(np.float32)

# Process image
out = ZNCC(img, part)

# Show and save image
cv2.namedWindow("result", 0)
cv2.resizeWindow("result", 256, 256)

cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("Myresult/out57.jpg", out)
