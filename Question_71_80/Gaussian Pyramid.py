# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt


def BGR2GRAY(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    out = 0.2126*r + 0.7152*g + 0.0722*b
    out = out.astype(np.uint8)

    return out

def bilinear_interpolation(img, ax=1, ay=1):
    
    if len(img.shape) > 2:
        H, W, C = img.shape
    else:
        H, W = img.shape
        C = 1

    aH = int(ay * H)
    aW = int(ax * W)

    # get position of resized image
    y = np.arange(aH).repeat(aW).reshape(aW, -1)
    x = np.tile(np.arange(aW), (aH, 1))

    # get position of original position
    y = (y / ay)
    x = (x / ax)

    ix = np.floor(x).astype(np.int)
    iy = np.floor(y).astype(np.int)

    ix = np.minimum(ix, W-2)
    iy = np.minimum(iy, H-2)

    # get distance 
    dx = x - ix
    dy = y - iy

    if C > 1:
        dx = np.repeat(np.expand_dims(dx, axis=-1), C, axis=-1)
        dy = np.repeat(np.expand_dims(dy, axis=-1), C, axis=-1)

    # interpolation
    out = (1-dx) * (1-dy) * img[iy, ix] + dx * (1 - dy) * img[iy, ix+1] + (1 - dx) * dy * img[iy+1, ix] + dx * dy * img[iy+1, ix+1]

    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out


# make image pyramid
def make_pyramid(gray):
	# first element
	pyramid = [gray]
	# each scale
	for i in range(1, 6):
		# define scale
		a = 2. ** i

		# down scale
		p = bilinear_interpolation(gray, ax=1./a, ay=1. / a)

		# up scale
		p = bilinear_interpolation(p, ax=a, ay=a)

		# add pyramid list
		pyramid.append(p.astype(np.float32))
		
	return pyramid

# make saliency map
def saliency_map(pyramid):
	# get shape
	H, W = pyramid[0].shape

	# prepare out image
	out = np.zeros((H, W), dtype=np.float32)

	# add each difference
	out += np.abs(pyramid[0] - pyramid[1])
	out += np.abs(pyramid[0] - pyramid[3])
	out += np.abs(pyramid[0] - pyramid[5])
	out += np.abs(pyramid[1] - pyramid[4])
	out += np.abs(pyramid[2] - pyramid[3])
	out += np.abs(pyramid[3] - pyramid[5])

	# normalization
	out = out / out.max() * 255

	return out

# Read image
img = cv2.imread("Jeanne.jpg").astype(np.float)

gray = BGR2GRAY(img)

# pyramid
pyramid = make_pyramid(gray)

for i in range(6):
    cv2.imwrite("Myresult/out75_{}.jpg".format(2**i), pyramid[i].astype(np.uint8))
    plt.subplot(1, 6, i+1)
    plt.imshow(pyramid[i], cmap='gray')
    plt.axis('off')
    plt.xticks(color="None")
    plt.yticks(color="None")

# plt.savefig("Myresult/out75.png")
plt.show()
