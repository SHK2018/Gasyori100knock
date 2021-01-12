import cv2
import numpy as np


# BGR -> HSV
def BGR2HSV(_img):
    img = _img.copy() / 255.

    hsv = np.zeros_like(img, dtype=np.float32)

    # get max and min
    max_v = np.max(img, axis=2).copy()
    min_v = np.min(img, axis=2).copy()
    min_arg = np.argmin(img, axis=2)

    # H
    hsv[..., 0][np.where(max_v == min_v)]= 0
    ## if min == B
    ind = np.where(min_arg == 0)
    hsv[..., 0][ind] = 60 * (img[..., 1][ind] - img[..., 2][ind]) / (max_v[ind] - min_v[ind]) + 60
    ## if min == R
    ind = np.where(min_arg == 2)
    hsv[..., 0][ind] = 60 * (img[..., 0][ind] - img[..., 1][ind]) / (max_v[ind] - min_v[ind]) + 180
    ## if min == G
    ind = np.where(min_arg == 1)
    hsv[..., 0][ind] = 60 * (img[..., 2][ind] - img[..., 0][ind]) / (max_v[ind] - min_v[ind]) + 300
        
    # S
    hsv[..., 1] = max_v.copy() - min_v.copy()

    # V
    hsv[..., 2] = max_v.copy()
    
    return hsv

def color_tracking(hsv, lt, ht):
    out = np.zeros_like(hsv[..., 0], dtype=np.float32)
    
    out[np.logical_and((hsv[..., 0] > lt), (hsv[..., 0] < ht))] = 255
    return out

# Read image
img = cv2.imread("Jeanne.jpg").astype(np.float32)

# BGR > HSV
hsv = BGR2HSV(img)

# Color Tracking
out = color_tracking(hsv, lt=25, ht=80).astype(np.uint8)

# Save result
cv2.imwrite("Myresult/out71.jpg", out)
cv2.namedWindow("result",0);
cv2.resizeWindow("result", 350, 350);
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()