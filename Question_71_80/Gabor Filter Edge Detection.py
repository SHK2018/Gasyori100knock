import cv2
import numpy as np
import matplotlib.pyplot as plt


# Grayscale
def BGR2GRAY(img):
	# Grayscale
	gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
	return gray

# Gabor
def Gabor_filter(K_size=11, Sigma=1.5, Gamma=1.2, Lambda=3, Psi=0, angle=0):
    # get half size
    d = K_size // 2

    # prepare kernel
    gabor = np.zeros((K_size, K_size), dtype=np.float32)

    # each value
    for y in range(K_size):
        for x in range(K_size):
            # distance from center
            kx = x - d
            ky = y - d

            # degree -> radian
            theta = angle / 180. * np.pi

            # get kernel x
            _x = np.cos(theta) * kx + np.sin(theta) * ky

            # get kernel y
            _y = -np.sin(theta) * kx + np.cos(theta) * ky

            # fill kernel
            gabor[y, x] = np.exp(-(_x**2 + Gamma**2 * _y**2) / (2 * Sigma**2)) * \
                         np.cos(2*np.pi*_x/Lambda + Psi)

    # kernel normalization
    gabor /= np.sum(np.abs(gabor))

    return gabor

def Gabor_filtering(gray, K_size=11, Sigma=1.5, Gamma=1.2, Lambda=3, Psi=0, angle=0):
    # get shape
    H, W = gray.shape

    # padding
    gray = np.pad(gray, (K_size//2, K_size//2), 'edge')

    # prepare out image
    out = np.zeros((H, W), dtype=np.float32)

    # get gabor filter
    gabor = Gabor_filter(K_size=K_size, Sigma=Sigma, Gamma=Gamma, Lambda=Lambda, Psi=0, angle=angle)
        
    # filtering
    for y in range(H):
        for x in range(W):
            out[y, x] = np.sum(gray[y : y + K_size, x : x + K_size] * gabor)
     
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out

# Read image
img = cv2.imread("Jeanne.jpg").astype(np.float32)

# gray scale
gray = BGR2GRAY(img).astype(np.float32)

for i in range(4):    
    out = Gabor_filtering(gray ,K_size=11, Sigma=1.5, Gamma=1.2, Lambda=3, Psi=0, angle=45*i)
    # cv2.imwrite("Myresult/out79_{}.jpg".format(45*i), out)
    plt.subplot(1, 4, i+1)
    plt.imshow(out, cmap='gray')
    plt.axis('off')
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.title("Angle " + str(45*i))

plt.savefig("Myresult/out79.png", dpi=326)
plt.show()
