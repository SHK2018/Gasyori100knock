import cv2
import numpy as np
import matplotlib.pyplot as plt


# Gabor
def Gabor_filter_rotated(K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=0):
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

for i in range(4):    
    # get gabor kernel
    gabor = Gabor_filter_rotated(K_size=111, Sigma=10, Gamma=1.2, Lambda=10, Psi=0, angle=45*i)
    
    # Visualize
    # normalize to [0, 255]
    out = gabor - np.min(gabor)
    out /= np.max(out)
    out *= 255
    
    plt.subplot(1, 4, i+1)
    plt.imshow(out, cmap='gray')
    plt.axis('off')
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.title("Angle " + str(45*i))

plt.savefig("Myresult/out78.png", dpi=326)
plt.show()
