# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt


def HOG_step3(img):
     # Grayscale
     def BGR2GRAY(img):
          gray = 0.2126 * img[..., 2] + 0.7152 * \
              img[..., 1] + 0.0722 * img[..., 0]
          return gray

     # Magnitude and gradient
     def get_gradXY(gray):
        H, W = gray.shape

        # padding before grad
        gray = np.pad(gray, (1, 1), 'edge')

        # get grad x
        gx = gray[1:H+1, 2:] - gray[1:H+1, :W]
        # get grad y
        gy = gray[2:, 1:W+1] - gray[:H, 1:W+1]
        # replace 0 with 
        gx[gx == 0] = 1e-6

        return gx, gy

     # get magnitude and gradient
     def get_MagGrad(gx, gy):
        # get gradient maginitude
        magnitude = np.sqrt(gx ** 2 + gy ** 2)

        # get gradient angle
        angle = np.arctan(gy / gx)

        angle[angle < 0] = np.pi / 2 + angle[angle < 0] + np.pi / 2

        return magnitude, angle

     # Gradient histogram
     def quantization(angle):
        # prepare quantization table
        angle_quantized = np.zeros_like(angle, dtype=np.int)

        # quantization base
        d = np.pi / 9

        # quantization
        for i in range(9):
            angle_quantized[np.where((angle >= d * i) & (angle <= d * (i + 1)))] = i

        return angle_quantized
      
     # get gradient histogram
     def gradient_histogram(angle_quantized, magnitude, N=8):
        # get shape
        H, W = magnitude.shape

        # get cell num
        cell_N_H = H // N
        cell_N_W = W // N
        histogram = np.zeros((cell_N_H, cell_N_W, 9), dtype=np.float32)

        # each pixel
        for y in range(cell_N_H):
            for x in range(cell_N_W):
                for j in range(N):
                    for i in range(N):
                        histogram[y, x, angle_quantized[y * N + j,
                                x * N]] += magnitude[y * N + j, x * N + i]

        return histogram
    
     # histogram normalization
     def normalization(histogram, C=3, epsilon=1):
        cell_N_H, cell_N_W, _ = histogram.shape
        ## each histogram
        for y in range(cell_N_H):
            for x in range(cell_N_W):
               #for i in range(9):
                histogram[y, x] /= np.sqrt(np.sum(histogram[max(y - 1, 0) : min(y + 2, cell_N_H),
                                                            max(x - 1, 0) : min(x + 2, cell_N_W)] ** 2) + epsilon)

        return histogram


     # 1. BGR -> Gray
     gray = BGR2GRAY(img)

     # 1. Gray -> Gradient x and y
     gx, gy = get_gradXY(gray)

     # 2. get gradient magnitude and angle
     magnitude, angle = get_MagGrad(gx, gy)

     # 3. Quantization
     angle_quantized = quantization(angle)
 
     # 4. Gradient histogram
     histogram = gradient_histogram(angle_quantized, magnitude)
     
     # 5. Histogram normalization
     histogram = normalization(histogram)
     
     return histogram


# Read image
img = cv2.imread("imori.jpg").astype(np.float32)

# get HOG step2
histogram = HOG_step3(img)
                
# write histogram to file
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(histogram[..., i])
    plt.axis('off')
    plt.xticks(color="None")
    plt.yticks(color="None")
plt.savefig("Myresult/out68.png")
plt.show()

