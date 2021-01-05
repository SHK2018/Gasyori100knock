# -*- coding: utf-8 -*-
import numpy as np
import cv2

# get IoU overlap ratio
def IoU(a, b):
	# get area of a
    area_a = (a[2] - a[0]) * (a[3] - a[1])
	# get area of b
    area_b = (b[2] - b[0]) * (b[3] - b[1])

	# get left top x of IoU
    iou_x1 = np.maximum(a[0], b[0])
	# get left top y of IoU
    iou_y1 = np.maximum(a[1], b[1])
	# get right bottom of IoU
    iou_x2 = np.minimum(a[2], b[2])
	# get right bottom of IoU
    iou_y2 = np.minimum(a[3], b[3])

	# get width of IoU
    iou_w = iou_x2 - iou_x1
	# get height of IoU
    iou_h = iou_y2 - iou_y1

	# no overlap
    if iou_w < 0 or iou_h < 0:
        return 0.0

	# get area of IoU
    area_iou = iou_w * iou_h
	# get overlap ratio between IoU and all area
    iou = area_iou / (area_a + area_b - area_iou)

    return iou

def crop_bbox(img, gt, Crop_N=200, L=60, th=0.5):
    H, W, C = img.shape
    
    for i in range(Crop_N):
        # get top letf x1 of crop bounding box
        x1 = np.random.randint(W - L)
        # get top letf y1 of crop bouding box
        y1 = np.random.randint(H - L)
        # get bottom right x2 and y2 of crop bounding box
        x2 = x1 + L
        y2 = y1 + L
        
        # crop bounding box
        crop = np.array((x1, y1, x2, y2))
        
        # get IoU between crop box and ground truth
        iou = IoU(gt, crop)
        
        # assian label
        if iou > th:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            label = 1
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
            label = 0
            
    return img

np.random.seed(0)

# read image
img = cv2.imread("imori_1.jpg")

# gt bounding box
gt = np.array((47, 41, 129, 103), dtype=np.float32)

# get crop bounding box
img = crop_bbox(img, gt)

# draw gt
cv2.rectangle(img, (gt[0], gt[1]), (gt[2], gt[3]), (0, 0, 255), 1)

cv2.imwrite("Myresult/out94.jpg", img)
cv2.imshow("result", img)
cv2.waitKey(0)