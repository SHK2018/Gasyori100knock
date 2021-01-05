# -*- coding: utf-8 -*-
import cv2
import numpy as np


def bilinear_interpolation(img, ax=1, ay=1):
    # input and out size
    H, W, C = img.shape
    aH = int(H * ay)
    aW = int(W * ax)

    out = np.zeros((aW, aH, C), dtype=np.float32)
    
    # i, j index for output image
    for i in range(aW):
        for j in range(aH):
            # yy, xx original index
            yy = round(j/ay)
            xx = round(i/ax)
            
            # weights base on distance
            dy = j/ay - yy
            dx = i/ax - xx

            if (yy+1<H) & (xx+1<W):
                out[j, i, :] = (1-dx)*(1-dy)*img[yy, xx, :] + dy*(1-dx)* \
                                img[yy+1, xx, :] + (1-dy)*dx*img[yy, xx+1\
                                , :] + dy*dx*img[yy+1, xx+1, :]
            elif(yy+1>=H) & (xx+1>=W):
                out[j, i, :] = (1-dx)*(1-dy)*img[yy, xx, :] + dy*(1-dx)* \
                                img[H-1, xx, :] + (1-dy)*dx*img[yy, W-1\
                                , :] + dy*dx*img[H-1, W-1, :]
            elif(xx+1>=W):
                out[j, i, :] = (1-dx)*(1-dy)*img[yy, xx, :] + dy*(1-dx)* \
                                img[yy+1, xx, :] + (1-dy)*dx*img[yy, W-1\
                                , :] + dy*dx*img[yy+1, W-1, :]
            elif(yy+1>=H):
                out[j, i, :] = (1-dx)*(1-dy)*img[yy, xx, :] + dy*(1-dx)* \
                                img[H-1, xx, :] + (1-dy)*dx*img[yy, xx+1\
                                , :] + dy*dx*img[H-1, xx+1, :]   
            
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


# Read image
img = cv2.imread("imori.jpg")

# Process image
out = bilinear_interpolation(img, 1.5, 1.5)

# Show and save result
cv2.namedWindow("result", 0)
cv2.resizeWindow("result", 256, 256)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.imwrite("Myresult/out26.jpg", out)
cv2.destroyAllWindows()
