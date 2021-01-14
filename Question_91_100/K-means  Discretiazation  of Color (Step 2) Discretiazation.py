# -*- coding: utf-8 -*-
import cv2
import numpy as np

# Dicrease color
def k_mean_dic_color(img, K=5):
    if len(img.shape) > 2:
        H, W, C = img.shape
    else:
        H, W = img.shape
        C = 1
    
    # reshape img into 2D
    tmp_img = img.reshape(H * W, C)
    
    # Step1: select one index randomly
    i = np.random.choice(np.arange(H * W), K, replace=False)
    color = tmp_img[i].copy()
    
    while True:
        # prepare pixel class label
        clss = np.zeros((H * W), dtype=int)
        
        # Step2: # get argmin distance for each pixel
        for i in range(H * W):
            # get distance from base pixel
            dis = np.sqrt(np.sum((color - tmp_img[i]) ** 2, axis=1))    
            clss[i] = np.argmin(dis)
            
        # Step3: Base on result generate new class
        color_tmp = np.zeros((K, 3))
        
        for i in range(K):
            color_tmp[i] = np.mean(tmp_img[clss == i], axis = 0)
        
        # if not any change
        if (color == color_tmp).all():
            break
        else:
            color = color_tmp.copy()
        
    # prepare out image
    out = np.zeros((H * W, 3), dtype=np.float32)

    # assign selected pixel values  
    for i in range(K):
        out[clss == i] = color[i]

    print(color)
        
    out = np.clip(out, 0, 255)

    # reshape out image
    out = np.reshape(out, (H, W, 3))
    out = out.astype(np.uint8)

    return out

# Read image
img = cv2.imread("Jeanne.jpg").astype(np.float32)

# Process image
out = k_mean_dic_color(img, K=10)

# Show and save image
cv2.namedWindow("result", 0)
cv2.resizeWindow("result", 512, 512)
cv2.imwrite("Myresult/out92.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()