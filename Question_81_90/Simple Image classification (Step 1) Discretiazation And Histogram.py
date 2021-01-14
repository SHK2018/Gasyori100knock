# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob


def decrease_color(img):
    out = img.copy()
    
    out = (out // 63) * 64 + 32
    
    return out

## Database
def get_DB():
    # get image paths
    train = glob("dataset/train_*") # Read all training data
    train.sort()
    
    # Set draw figure size
    plt.figure(figsize=(19.20, 10.80))
    
    # prepare database
    db = np.zeros((len(train), 13), dtype=np.int32) # 13 = (B + G + R) * 4 + tag

    # each image
    for i, path in enumerate(train):
        img = decrease_color(cv2.imread(path))
        # get histogram
        for j in range(4):
            # count for numbers of pixels
            db[i, j] = len(np.where(img[..., 0] == (64 * j + 32))[0])
            db[i, j+4] = len(np.where(img[..., 1] == (64 * j + 32))[0])
            db[i, j+8] = len(np.where(img[..., 2] == (64 * j + 32))[0])

        # get class
        if 'akahara' in path:
            cls = 0
        elif 'madara' in path:
            cls = 1

        # store class label
        db[i, -1] = cls
        
        # for histogram: B(1,4), B(5,8), B(9,12)
        img_h = img.copy() // 64
        img_h[..., 1] += 4
        img_h[..., 2] += 8
        
        plt.subplot(2, 5, i+1)
        plt.hist(img_h.ravel(), bins=12, rwidth=0.8)
        plt.title(path[15:])

    print(db)
    plt.savefig("Myresult/out84.png", dpi=326)
    plt.show()
    
    return db

# get database
database = get_DB()