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


def get_DB(dataset):
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

    return db


def predict(trainDB, testDB):
    # [idx_in_folder, class]
    results = []

    for i in range(len(testDB)):
        # get differences between one test_img and all train_imgs
        df = trainDB[:, :-1] - testDB[i][:-1]
        # get min differences index
        df_sum = np.abs(df).sum(axis=1)
        df_min_idx = np.where(df_sum == np.min(df_sum))
        results.append([int(df_min_idx[0]), int(
            trainDB[df_min_idx][..., -1][0])])

    return results


def print_result(result, trainpath, testpath):
    # class
    tag = ["akahara", "madara"]

    traindata = glob(trainpath)  # Read all training data
    traindata.sort()

    testdata = glob(testpath)  # Read all training data
    testdata.sort()

    for i in range(len(result)):
        print("%s is similar >> %s  Pred >> %s" %
              (testdata[i], traindata[result[i][0]], tag[result[i][1]]))


# get database
trainpath = "dataset/train_*"
testpath = "dataset/test_*"

trainDB = get_DB(trainpath)
testDB = get_DB(testpath)

# predict test image
result = predict(trainDB, testDB)

# print result
print_result(result, trainpath, testpath)
