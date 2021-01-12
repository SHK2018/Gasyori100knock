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
    color = np.zeros_like(hsv[..., 0], dtype=np.float)
    out = hsv.copy()
    
    color[np.logical_and((hsv[..., 0] > lt), (hsv[..., 0] < ht))] = 1
     
    # closing
    color = Morphology_Closing(color, time=5)
    
    # opening
    color = Morphology_Opening(color, time=5)
    
    out[color == 1, :] = 0
    return out

def HSV2BGR(_img, hsv):
	img = _img.copy() / 255.

	# get max and min
	max_v = np.max(img, axis=2).copy()
	min_v = np.min(img, axis=2).copy()

	out = np.zeros_like(img)

	H = hsv[..., 0]
	S = hsv[..., 1]
	V = hsv[..., 2]

	C = S
	H_ = H / 60.
	X = C * (1 - np.abs( H_ % 2 - 1))
	Z = np.zeros_like(H)
	vals = [[Z,X,C], [Z,C,X], [X,C,Z], [C,X,Z], [C,Z,X], [X,Z,C]]

	for i in range(6):
		ind = np.where((i <= H_) & (H_ < (i+1)))
		out[..., 0][ind] = (V - C)[ind] + vals[i][0][ind]
		out[..., 1][ind] = (V - C)[ind] + vals[i][1][ind]
		out[..., 2][ind] = (V - C)[ind] + vals[i][2][ind]

	out[np.where(max_v == min_v)] = 0
	out = np.clip(out, 0, 1)
	out = (out * 255).astype(np.uint8)

	return out

# Erosion
def Erode(img, Erode_time=1):
	H, W = img.shape
	out = img.copy()

	# kernel
	MF = np.array(((0, 1, 0),
				(1, 0, 1),
				(0, 1, 0)), dtype=np.int)

	# each erode
	for i in range(Erode_time):
		tmp = np.pad(out, (1, 1), 'edge')
		# erode
		for y in range(1, H + 1):
			for x in range(1, W + 1):
				if np.sum(MF * tmp[y - 1 : y + 2 , x - 1 : x + 2]) < 1 * 4:
					out[y - 1, x - 1] = 0

	return out


# Dilation
def Dilate(img, Dil_time=1):
	H, W = img.shape

	# kernel
	MF = np.array(((0, 1, 0),
				(1, 0, 1),
				(0, 1, 0)), dtype=np.int)

	# each dilate time
	out = img.copy()
	for i in range(Dil_time):
		tmp = np.pad(out, (1, 1), 'edge')
		for y in range(1, H + 1):
			for x in range(1, W + 1):
				if np.sum(MF * tmp[y - 1 : y + 2, x - 1 : x + 2]) >= 1:
					out[y - 1, x - 1] = 1

	return out


# Opening morphology
def Morphology_Opening(img, time=1):
    out = Erode(img, Erode_time=time)
    out = Dilate(out, Dil_time=time)
    return out

# Closing morphology
def Morphology_Closing(img, time=1):
	out = Dilate(img, Dil_time=time)
	out = Erode(out, Erode_time=time)
	return out


# Read image
img = cv2.imread("Jeanne.jpg").astype(np.float32)

# BGR > HSV
hsv = BGR2HSV(img)

# Color Tracking
color = color_tracking(hsv, lt=25, ht=80)

# HSV > BGR
out = HSV2BGR(img, color)

# Save result
cv2.imwrite("Myresult/out72.jpg", out)
cv2.namedWindow("result",0);
cv2.resizeWindow("result", 512, 512);
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()