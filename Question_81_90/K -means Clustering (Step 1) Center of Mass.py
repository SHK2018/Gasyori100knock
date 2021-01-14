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

    return db

def k_means_step2(db, th=0.5):
    tmp_db = db.copy()
    
    for i in range(len(tmp_db)):
        # get random label
        label = 0
        if np.random.random(1,)[0] > th:
            label = 1
        
        tmp_db[i, -1] = label
        
    # get grabity for each class
    grabity = []
    
    for i in range(2):
        # img in this class
        imgs = tmp_db[np.where(tmp_db[...,-1] == i)][:,:-1]
        grabity.append(sum(imgs) / len(imgs))
        print("The grabity for class %d is: " % i, grabity[-1])
    
    return grabity


def predict(trainDB, testDB):
    # [idx_in_folder, class]
    result = []

    for i in range(len(testDB)):
        # get differences between one test_img and all train_imgs
        df = trainDB[:, :-1] - testDB[i][:-1]
        # get min differences index
        df_sum = np.abs(df).sum(axis=1)
        df_min_idx = np.argsort(df_sum)[:3]
        result.append([df_min_idx, trainDB[df_min_idx][..., -1]])

    return result


def print_result(result, trainpath, testpath):
    # class
    label = ["akahara", "madara"]

    traindata = glob(trainpath)  # Read all training data
    traindata.sort()

    testdata = glob(testpath)  # Read all training data
    testdata.sort()

    for i in range(len(result)):
        print("%s is similar >> %s,  %s,  %s  | Pred >> %s" % (testdata[i], 
            traindata[result[i][0][0]], traindata[result[i][0][1]], 
            traindata[result[i][0][2]], label[np.argmax(np.bincount(result[i][1]))]))
        
def validate(result, trainDB, testDB):

    sum = 0
    
    for i in range(len(result)):    
        if np.argmax(np.bincount(result[i][1])) == int(testDB[i][..., -1]):
            sum +=1
        
    accuracy = sum / len(result)*100
        
    print("Accuracy >> %6.2f" % accuracy)
    


# get database
trainpath = "dataset/train_*"
testpath = "dataset/test_*"

# trainDB = get_DB(trainpath)
testDB = get_DB(testpath)

# get grabity
grabity = k_means_step2(testDB)

# # predict test image
# result = predict(trainDB, testDB)

# # print result
# print_result(result, trainpath, testpath)

# # # validat reuslt
# validate(result, trainDB, testDB)
