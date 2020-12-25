# -*- coding: utf-8 -*-
import cv2
import numpy as np


def BGR2HSV(img):
    _img = img.copy() / 255.0
    
    hsv = np.zeros_like(_img, dtype=np.float32)
    
    # Max and Min value among three channels
    max_v = np.max(_img, axis=2).copy()
    min_v = np.min(_img, axis=2).copy()
    # Min value's channel index
    min_arg = np.argmin(_img, axis=2)
    #HSV H
    ## Slice, ... which keep the length of turple corresponds with the length 
    ## of array
    ## if max = min
    hsv[..., 0][np.where(max_v == min_v)] = 0
    ## if min = B
    idx = np.where(min_arg == 0)
    hsv[..., 0][idx] = 60 * (_img[:,:,1][idx] - _img[:, :, 2][idx]) / (max_v[idx] - min_v[idx]) \
        + 60
    ## if min = R
    idx = np.where(min_arg == 1)
    hsv[..., 0][idx] = 60 * (_img[:,:,0][idx] - _img[:, :, 1][idx]) / (max_v[idx] - min_v[idx]) \
        + 180
    ## if min = G
    idx = np.where(min_arg == 2)
    hsv[..., 0][idx] = 60 * (_img[:,:,2][idx] - _img[:, :, 0][idx]) / (max_v[idx] - min_v[idx]) \
        + 300
        
    # S and V
    hsv[..., 2] = max_v.copy()
    hsv[..., 1] = max_v.copy() - min_v.copy()
        
    return hsv

def HSV2RGB(img, hsv):
	img = _img.copy() / 255.

	# get max and min
	max_v = np.max(img, axis=2).copy()
	min_v = np.min(img, axis=2).copy()

	out = np.zeros_like(img)
    
    H = hsv[..., 0] / 60
    C = hsv[..., 1]
    X = C * (1 - abs(H % 2 -1))
    
    
# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# RGB2HSV
hsv = BGR2HSV(img)

# Transpose Hue
hsv[..., 0] = (hsv[..., 0] + 180) % 360
    
# HSV2RGB
out = HSV2RGB(img, hsv)