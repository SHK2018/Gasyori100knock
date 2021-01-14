# -*- coding: utf-8 -*-
import cv2
import numpy as np

# Dicrease color
def k_mean_dic_color_step1(img, K=5):
    if len(img.shape) > 2:
        H, W, C = img.shape
    else:
        H, W = img.shape
        C = 1
    
    # reshape img into 2D
    tmp_img = img.reshape(H * W, C)
    
    # select one index randomly
    i = np.random.choice(np.arange(H * W), K, replace=False)
    color = tmp_img[i].copy()
    
    print(color)
    
    clss = np.zeros((H * W), dtype=int)
    
    # each pixel
    for i in range(H * W):
        # get distance from base pixel
        dis = np.sqrt(np.sum((color - tmp_img[i]) ** 2, axis=1))
        # get argmin distance
        clss[i] = np.argmin(dis)

    # show
    out = np.reshape(clss, (H, W)) * 50
    out = out.astype(np.uint8)

    return out

# Read image
img = cv2.imread("Jeanne.jpg").astype(np.float32)

# Process image
out = k_mean_dic_color_step1(img)

# Show and save image
cv2.namedWindow("result", 0)
cv2.resizeWindow("result", 512, 512)
cv2.imwrite("Myresult/out91.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()