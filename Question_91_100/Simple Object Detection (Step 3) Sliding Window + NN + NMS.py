# -*- coding: utf-8 -*-
import numpy as np
import cv2
# import matplotlib.pyplot as plt

## Database
def get_db(img, gt, N=200, size=32, L=60, th=0.5):
    
    # Set draw figure size
    # plt.figure(figsize=(19.20, 10.80))
    
    # get HOG feature dimension
    HOG_feature_N = ((size // 8) ** 2) * 9
    
    # prepare database
    label = np.zeros([N, 1], dtype=np.uint8)
    db = np.zeros([N, HOG_feature_N])

    # each image
    for i in range(N):
        # get bounding box
        cropped_img, cropped_label = crop_bbox(img, gt, 1, L=L, th=th)
        
        # get HOG feature
        hog = HOG(cropped_img)
        
        # store HOG feature and label
        db[i, :HOG_feature_N] = hog.ravel()
        # save coresponding label
        label[i, :] = cropped_label
        
        # for histogram: B(1,4), B(5,8), B(9,12)
        # img_h = cropped_img.copy() // 64
        # img_h[..., 1] += 4
        # img_h[..., 2] += 8
        
    #     plt.subplot(2, N/2, i+1)
    #     plt.hist(img_h.ravel(), bins=12, rwidth=0.8)
    #     plt.title(i)

    # plt.show()
    
    return db, label

def crop_bbox(img, gt, Crop_N=200, L=60, th=0.5):
    H, W, C = img.shape
    
    for i in range(Crop_N):
        # get top letf x1 of crop bounding box
        x1 = np.random.randint(W - L)
        # get top letf y1 of crop bouding box
        y1 = np.random.randint(H - L)
        # get bottom right x2 and y2 of crop bounding box
        x2 = x1 + L
        y2 = y1 + L
        
        # crop bounding box
        crop = np.array((x1, y1, x2, y2))
        
        # get IoU between crop box and ground truth
        iou = IoU(gt, crop)
        
        # crop training data and assian label
        if iou > th:
            train_img = cv2.resize(img[y1:y2, x1:x2], (32, 32))
            label = 1 
        else:
            train_img = cv2.resize(img[y1:y2, x1:x2], (32, 32))
            label = 0
            
    return train_img, label


# neural network
class NN:
    def __init__(self, ind=2, w=64, w2=64, outd=1, lr=0.1):
        # layer 1 weight
        self.w1 = np.random.normal(0, 1, [ind, w])
        # layer 1 bias
        self.b1 = np.random.normal(0, 1, [w])
        # layer 2 weight
        self.w2 = np.random.normal(0, 1, [w, w2])
        # layer 2 bias
        self.b2 = np.random.normal(0, 1, [w2])
        # output layer weight
        self.wout = np.random.normal(0, 1, [w2, outd])
        # output layer bias
        self.bout = np.random.normal(0, 1, [outd])
        # learning rate
        self.lr = lr

    def forward(self, x):
        # input tensor
        self.z1 = x
        # layer 1 output tensor
        self.z2 = sigmoid(np.dot(self.z1, self.w1) + self.b1)
        # layer 2 output tensor
        self.z3 = sigmoid(np.dot(self.z2, self.w2) + self.b2)
        # output layer tensor
        self.out = sigmoid(np.dot(self.z3, self.wout) + self.bout)
        return self.out

    def train(self, x, t):
        # backpropagation output layer
        #En = t * np.log(self.out) + (1-t) * np.log(1-self.out)
        En = (self.out - t) * self.out * (1 - self.out)
        # get gradients for weight and bias
        grad_wout = np.dot(self.z3.T, En)
        grad_bout = np.dot(np.ones([En.shape[0]]), En)
        # update weight and bias
        self.wout -= self.lr * grad_wout
        self.bout -= self.lr * grad_bout

        # backpropagation inter layer
        # get gradients for weight and bias
        grad_u2 = np.dot(En, self.wout.T) * self.z3 * (1 - self.z3)
        grad_w2 = np.dot(self.z2.T, grad_u2)
        grad_b2 = np.dot(np.ones([grad_u2.shape[0]]), grad_u2)
        # update weight and bias
        self.w2 -= self.lr * grad_w2
        self.b2 -= self.lr * grad_b2
        
        # get gradients for weight and bias
        grad_u1 = np.dot(grad_u2, self.w2.T) * self.z2 * (1 - self.z2)
        grad_w1 = np.dot(self.z1.T, grad_u1)
        grad_b1 = np.dot(np.ones([grad_u1.shape[0]]), grad_u1)
        # update weight and bias
        self.w1 -= self.lr * grad_w1
        self.b1 -= self.lr * grad_b1

# sigmoid
def sigmoid(x):
    return 1. / (1. + np.exp(-x))

# train
def train_nn(nn, train_x, train_t, iteration_N=5000):
    for i in range(5000):
        # feed-forward data
        nn.forward(train_x)
        #print("ite>>", i, 'y >>', nn.forward(train_x))
        # update parameters
        nn.train(train_x, train_t)

    return nn


# test
def test_nn(nn, test_x, test_t, pred_th=0.5):
    accuracy_N = 0.

    # each data
    for data, t in zip(test_x, test_t):
        # get prediction
        prob = nn.forward(data)

        # count accuracy
        pred = 1 if prob >= pred_th else 0
        if t == pred:
            accuracy_N += 1

    # get accuracy 
    accuracy = accuracy_N / len(test_x)

    print("Accuracy >> {} ({} / {})".format(accuracy, accuracy_N, len(test_x)))

# get IoU overlap ratio
def IoU(a, b):
    # get area of a
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    # get area of b
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    # get left top x of IoU
    iou_x1 = np.maximum(a[0], b[0])
    # get left top y of IoU
    iou_y1 = np.maximum(a[1], b[1])
    # get right bottom of IoU
    iou_x2 = np.minimum(a[2], b[2])
    # get right bottom of IoU
    iou_y2 = np.minimum(a[3], b[3])

    # get width of IoU
    iou_w = iou_x2 - iou_x1
    # get height of IoU
    iou_h = iou_y2 - iou_y1

    # no overlap
    if iou_w < 0 or iou_h < 0:
        return 0.0

    # get area of IoU
    area_iou = iou_w * iou_h
    # get overlap ratio between IoU and all area
    iou = area_iou / (area_a + area_b - area_iou)

    return iou

# get HOG
def HOG(img):
    # Grayscale
    def BGR2GRAY(img):
        gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
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
        gradient = np.arctan(gy / gx)

        gradient[gradient < 0] = np.pi / 2 + gradient[gradient < 0] + np.pi / 2

        return magnitude, gradient

    # Gradient histogram
    def quantization(gradient):
        # prepare quantization table
        gradient_quantized = np.zeros_like(gradient, dtype=np.int)

        # quantization base
        d = np.pi / 9

        # quantization
        for i in range(9):
            gradient_quantized[np.where((gradient >= d * i) & (gradient <= d * (i + 1)))] = i

        return gradient_quantized


    # get gradient histogram
    def gradient_histogram(gradient_quantized, magnitude, N=8):
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
                        histogram[y, x, gradient_quantized[y * 4 + j, x * 4 + i]] += magnitude[y * 4 + j, x * 4 + i]

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
    magnitude, gradient = get_MagGrad(gx, gy)

    # 3. Quantization
    gradient_quantized = quantization(gradient)

    # 4. Gradient histogram
    histogram = gradient_histogram(gradient_quantized, magnitude)
    
    # 5. Histogram normalization
    histogram = normalization(histogram)

    return histogram

## Database
def sliding_win_NN(img, size=32, stride=4, th=0.5):
    H, W, C = img.shape
    
    detects = np.ndarray((0, 5), dtype=np.float32)
    
    # base rectangle [h, w]
    recs = np.array(((42, 42), (56, 56), (70, 70)), dtype=np.float32)
    
    # sliding window
    for rect in recs:
        for y in range(0, H, stride):
            for x in range(0, W, stride):

                # get half size of ractangle
                dh = int(rect[0] // 2)
                dw = int(rect[1] // 2)

                # get left top x
                x1 = max(x - dw, 0)
                # get left top y
                x2 = min(x + dw, W)
                # get right bottom x
                y1 = max(y - dh, 0)
                # get right bottom y
                y2 = min(y + dh, H)

                # crop region
                region = img[y1 : y2, x1 : x2]

                # resize crop region
                region = cv2.resize(region, (size, size))

                # get HOG feature
                region_hog = HOG(region).ravel()
                
                # apply NN to do prediction
                prob = nn.forward(region_hog)

                # count accuracy
                pred = 1 if prob >= th else 0
                
                if pred:
                    detects = np.vstack((detects, np.array((x1, y1, x2, y2, prob))))
                
    return detects

# Non-maximum suppression
def nms(_bboxes, iou_th=0.5, select_num=None, prob_th=None):
    #
    # Non Maximum Suppression
    #
    # Argument
    #  bboxes(Nx5) ... [bbox-num, 5(leftTopX,leftTopY,w,h, score)]
    #  iou_th([float]) ... threshold for iou between bboxes.
    #  select_num([int]) ... max number for choice bboxes. If None, this is unvalid.
    #  prob_th([float]) ... probability threshold to choice. If None, this is unvalid.
    # Return
    #  inds ... choced indices for bboxes
    #

    bboxes = _bboxes.copy()
    
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    
    # Sort by bbox's score. High -> Low
    sort_inds = np.argsort(bboxes[:, -1])[::-1]

    processed_bbox_ind = []
    return_inds = []

    unselected_inds = sort_inds.copy()
    
    while len(unselected_inds) > 0:
        process_bboxes = bboxes[unselected_inds]
        argmax_score_ind = np.argmax(process_bboxes[::, -1])
        max_score_ind = unselected_inds[argmax_score_ind]
        return_inds += [max_score_ind]
        unselected_inds = np.delete(unselected_inds, argmax_score_ind)

        base_bbox = bboxes[max_score_ind]
        compare_bboxes = bboxes[unselected_inds]
        
        base_x1 = base_bbox[0]
        base_y1 = base_bbox[1]
        base_x2 = base_bbox[2] + base_x1
        base_y2 = base_bbox[3] + base_y1
        base_w = np.maximum(base_bbox[2], 0)
        base_h = np.maximum(base_bbox[3], 0)
        base_area = base_w * base_h

        # compute iou-area between base bbox and other bboxes
        iou_x1 = np.maximum(base_x1, compare_bboxes[:, 0])
        iou_y1 = np.maximum(base_y1, compare_bboxes[:, 1])
        iou_x2 = np.minimum(base_x2, compare_bboxes[:, 2] + compare_bboxes[:, 0])
        iou_y2 = np.minimum(base_y2, compare_bboxes[:, 3] + compare_bboxes[:, 1])
        iou_w = np.maximum(iou_x2 - iou_x1, 0)
        iou_h = np.maximum(iou_y2 - iou_y1, 0)
        iou_area = iou_w * iou_h

        compare_w = np.maximum(compare_bboxes[:, 2], 0)
        compare_h = np.maximum(compare_bboxes[:, 3], 0)
        compare_area = compare_w * compare_h

        # bbox's index which iou ratio over threshold is excluded
        all_area = compare_area + base_area - iou_area
        iou_ratio = np.zeros((len(unselected_inds)))
        iou_ratio[all_area < 0.9] = 0.
        _ind = all_area >= 0.9
        iou_ratio[_ind] = iou_area[_ind] / all_area[_ind]
        
        unselected_inds = np.delete(unselected_inds, np.where(iou_ratio >= iou_th)[0])

    if prob_th is not None:
        preds = bboxes[return_inds][:, -1]
        return_inds = np.array(return_inds)[np.where(preds >= prob_th)[0]].tolist()
        
    # pick bbox's index by defined number with higher score
    if select_num is not None:
        return_inds = return_inds[:select_num]

    return return_inds

# %% Preparing neural network
# read image
img1 = cv2.imread("imori_1.jpg")

# gt bounding box
gt = np.array((47, 41, 129, 103), dtype=np.float32)

# train data and label data
train_x, train_t = get_db(img1, gt, th=0.5)

# # train data and label data
# test_x, test_t = get_db(img1, gt, 100, th=0.5)

# prepare neural network
nn = NN(train_x.shape[1], lr=0.01)

# train
nn = train_nn(nn, train_x, train_t, iteration_N=10000)

# # test
# test_nn(nn, test_x, test_t)

# %% sliding windows and NN detection
# read image
img2 = cv2.imread("imori_many.jpg")

detects = sliding_win_NN(img2, th=0.7)
detects = detects[nms(detects, iou_th=0.25)]

for d in detects:
    v = list(map(int, d[:4]))
    cv2.rectangle(img2, (v[0], v[1]), (v[2], v[3]), (0,0,255), 1)
    cv2.putText(img2, "{:.2f}".format(d[-1][0]), (v[0], v[1]+9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255), 1)

cv2.imwrite("Myresult/out99.jpg", img2)
cv2.imshow("result", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
