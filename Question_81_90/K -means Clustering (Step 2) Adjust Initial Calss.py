# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob


def decrease_color(img):
    out = img.copy()

    out = (out // 63) * 64 + 32

    return out

# Database


def get_DB(dataset, th=0.5):
    # get image paths
    data = glob(dataset)  # Read all training data
    data.sort()

    # Set draw figure size
    plt.figure(figsize=(19.20, 10.80))

    # prepare database
    # 13 = (B + G + R) * 4 + tag
    db = np.zeros((len(data), 13), dtype=np.int32)

    # each image
    for i, path in enumerate(data):
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

        plt.subplot(2, int(len(data)/2), i+1)
        plt.hist(img_h.ravel(), bins=12, rwidth=0.8)
        plt.title(path[15:])

    # print(db)
    # plt.savefig("Myresult/out84.png", dpi=326)
    plt.show()

    return db, data

def k_means_step3(db, pdata, th=0.5, Class=2):
    tmp_db = db.copy()
    
    # initiate random seed
    np.random.seed(4)
    
    for i in range(len(tmp_db)):
        # get random label
        label = 0
        if np.random.random(1,)[0] > th:
            label = 1
        
        tmp_db[i, -1] = label
    while True:
        num = 0
        # get grabity for each class
        grabity = np.zeros((2, len(tmp_db[0])-1), dtype=np.float32)
        
        for i in range(2):
            grabity[i] = np.mean(tmp_db[np.where(tmp_db[..., -1] == i)[0], :len(tmp_db[0])-1], axis=0)
        
        for i in range(len(tmp_db)):
            # get distance each nearest graviry
            dis = np.sqrt(np.sum(np.square(np.abs(grabity - tmp_db[i, :len(tmp_db[0])-1])), axis=1))
            # get new label
            pred = np.argmin(dis, axis=0)

            # if label is difference from old label
            if int(tmp_db[i, -1]) != pred:
                num += 1
                tmp_db[i, -1] = pred
        
        if num < 1:
            break
    
    for i in range(db.shape[0]):
        print(pdata[i], " Pred:", tmp_db[i, -1])


# get database
trainpath = "dataset/train_*"

trainDB, pdata = get_DB(trainpath)

# clustering
reuslt = k_means_step3(trainDB, pdata, th=0.3)
