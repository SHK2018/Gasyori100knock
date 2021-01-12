# -*- coding: utf-8 -*-
import cv2
import numpy as np


def HOG_step1(img):
     # Grayscale
     def BGR2GRAY(img):
          gray = 0.2126 * img[..., 2] + 0.7152 * \
              img[..., 1] + 0.0722 * img[..., 0]
          return gray

     # Magnitude and gradient
     def get_gradXY(gray):
          H, W = gray.shape

          # padding before grad
          gray = np.pad(gray, (1,1), 'edge')

          # get grad x
          gx = gray[1:H+1, 2:] - gray[1:H+1, :W]
          # get grad y
          gy = gray[2:, 1:W+1] - gray[:H, 1:W+1]
          # replace 0 with 1e-6
          gx[gx == 0] = 1e-6

          return gx, gy

     # get magnitude and gradient
     def get_MagGrad(gx, gy):
          # get gradient maginitude
          magnitude = np.sqrt(gx**2 + gy**2)

          # get gradient angle
          angle = np.arctan(gy / gx)

          return magnitude, angle

     # Gradient histogram
     def quantization(angle):
          # prepare quantization table
          angle_quantized = np.zeros_like(angle, dtype = int)

          # quantization base
          d = np.pi / 9

          # quantization
          for i in range(9):
              angle_quantized[np.where((angle>d*i) & (angle<=d*(i+1)))] = i

          return angle_quantized

     # 1. BGR -> Gray
     gray = BGR2GRAY(img)

     # 1. Gray -> Gradient x and y
     gx, gy = get_gradXY(gray)

     # 2. get gradient magnitude and angle
     magnitude, angle = get_MagGrad(gx, gy)

     # 3. Quantization
     angle_quantized = quantization(angle)

     return magnitude, angle_quantized


# Read image
img = cv2.imread("imori.jpg")

# Process image
magnitude, angle_quantized = HOG_step1(img)

# Gradient magnitude
_magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)

# Write gradient angle to file
H, W, C = img.shape
out = np.zeros((H, W, 3), dtype=np.uint8)

# define color
C = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255],
     [127, 127, 0], [127, 0, 127], [0, 127, 127]]

# draw color
for i in range(9):
     out[angle_quantized == i] = C[i]

_magnitude = np.expand_dims(_magnitude, 2).repeat(3,axis=2)
output = np.hstack([_magnitude, out])
cv2.imwrite("Myresult/out66.jpg", output)

cv2.namedWindow("result", 0)
cv2.resizeWindow("result", 512,256)

cv2.imshow("result", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

