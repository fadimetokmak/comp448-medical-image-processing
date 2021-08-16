
import numpy as np
import math
from PIL import Image
from numpy import asarray
from skimage.color import rgb2gray
from sklearn import svm

##PART 1--------------------------------------
def group(arr, binNumber):
    max = np.max(arr)
    min = np.min(arr)
    ran = max - min
    lst = []
    for i in range(1, binNumber):
        lst.append(math.floor(i * (ran / binNumber)))
    lst.append(max + 1)
    return lst


def find_group(groups, num):
    count = 0
    for group in groups:
        if group > num:
            return count
        count += 1


def calculateCooccurrenceMatrix(gray_image, bin_number, di, dj):
    im = gray_image * 256
    groups = group(im, bin_number)
    co_occurence = np.zeros((bin_number, bin_number))
    for i in range(len(im)):
        for j in range(len(im[0])):
            new_index1 = i + di
            new_index2 = j + dj
            if new_index1 < 0 or new_index1 >= len(im) or new_index2 < 0 or new_index2 >= len(im[0]):
                continue
            f = find_group(groups, im[i, j])
            s = find_group(groups, im[new_index1, new_index2])
            co_occurence[int(f), int(s)] = co_occurence[int(f), int(s)] + 1
    return co_occurence

def calculateAccumulatedCooccurrenceMatrix(grayImg, binNumber, d):
    result = np.zeros((binNumber,binNumber))
    result += calculateCooccurrenceMatrix(grayImg, binNumber, d, 0)
    result += calculateCooccurrenceMatrix(grayImg, binNumber, d, d)
    result += calculateCooccurrenceMatrix(grayImg, binNumber, 0, d)
    result += calculateCooccurrenceMatrix(grayImg, binNumber, -d, d)
    result += calculateCooccurrenceMatrix(grayImg, binNumber, -d, 0)
    result += calculateCooccurrenceMatrix(grayImg, binNumber, -d, -d)
    result += calculateCooccurrenceMatrix(grayImg, binNumber, 0, -d)
    result += calculateCooccurrenceMatrix(grayImg, binNumber, d, -d)
    return result

def calculateCooccurrenceFeatures(accM):
    normalized = accM/accM.sum()
    sums_x =[]
    sums_y = []
    for i in range(len(accM)):
        sums_x.append(accM[i,:].sum())
    for i in range(len(accM)):
        sums_y.append(accM[:,i].sum())
    angular_sc_moment = np.square(normalized).sum()
    max_prob = normalized.max()
    inv_dif_movement = 0
    contrast = 0
    entropy = 0
    correlation = 0
    for i in range(len(accM)):
        for j in range(len(accM)):
            inv_dif_movement += accM[i,j]/(1+ (i-j)*(i-j))
            contrast += (i-j)*(i-j)*accM[i,j]
            if(accM[i,j] != 0):
                entropy -= accM[i,j]*math.log(accM[i,j])
            correlation += i*j*accM[i,j]
    correlation = (correlation - np.mean(np.array(sums_x))*np.mean(np.array(sums_y))/(np.std(np.array(sums_x))*np.std(np.array(sums_y))))
    return angular_sc_moment, max_prob, inv_dif_movement, contrast, entropy, correlation

#PART2------------------------------------
def PART2():
    # feature extraction from training dataset
    labels = np.loadtxt("labels.txt") #I copied images for balancing dataset and I produce a new labels file according to copying
    features = []
    for i in range(1,267):
        string = "tr" + str(i) + ".jpg"
        image = Image.open(string)
        data = asarray(image)
        im = rgb2gray(data)
        i1, i2, i3, i4, i5, i6 = calculateCooccurrenceFeatures(calculateAccumulatedCooccurrenceMatrix(im,8,10))
        features.append((i1,i2,i3,i4,i5,i6,labels[i-1]))

    ## Normalization of Training Features
    training_feature_array = np.array(features)
    for i in (range(len(training_feature_array[0]) - 1)):
        mean = np.mean(training_feature_array[:, i])
        dev = np.std(training_feature_array[:, i])
        for j in range(len(training_feature_array)):
            training_feature_array[j, i] = (training_feature_array[j, i] - mean) / dev

    # feature extraction from test dataset
    test_labels = np.loadtxt("test_labels.txt")
    test_features = []
    for i in range(1, 144):
        string = "ts" + str(i) + ".jpg"
        image = Image.open(string)
        data = asarray(image)
        im = rgb2gray(data)
        i1, i2, i3, i4, i5, i6 = calculateCooccurrenceFeatures(calculateAccumulatedCooccurrenceMatrix(im, 8, 10))
        test_features.append((i1, i2, i3, i4, i5, i6, test_labels[i - 1]))

    ## Normalization of Test Features
    test_feature_array = np.array(test_features)
    for i in (range(len(test_feature_array[0]) - 1)):
        mean = np.mean(test_feature_array[:, i])
        dev = np.std(test_feature_array[:, i])
        for j in range(len(test_feature_array)):
            test_feature_array[j, i] = (test_feature_array[j, i] - mean) / dev

    #Training
    linear_kernel = svm.SVC(C=10, kernel='linear')
    linear_kernel.fit(training_feature_array[:, :6], training_feature_array[:, 6])
    linear_pred = linear_kernel.predict(test_feature_array[:, :6])
    rbf_kernel = svm.SVC(C=11500, kernel='rbf')
    rbf_kernel.fit(training_feature_array[:, :6], training_feature_array[:, 6])
    rbf_pred = rbf_kernel.predict(test_feature_array[:, :6])

    return linear_pred, rbf_pred

#PART 3----------------------------------------------------------------------
def grid_features(im, N, binNumber, d):
    f1, f2, f3, f4, f5, f6 = 0, 0, 0, 0, 0, 0
    lenx = len(im)
    leny = len(im[0])
    for i in range(N):
        for j in range(N):
            grid = im[int(lenx * i / N):int(lenx * (i + 1) / N), int(leny * j / N):int(leny * (j + 1) / N)]
            x1, x2, x3, x4, x5, x6 = calculateCooccurrenceFeatures(
                calculateAccumulatedCooccurrenceMatrix(grid, binNumber, d))
            f1 += x1
            f2 += x2
            f3 += x3
            f4 += x4
            f5 += x5
            f6 += x6
    return f1 / (N * N), f2 / (N * N), f3 / (N * N), f4 / (N * N), f5 / (N * N), f6 / (N * N)


def Part3():
    # feature extraction from training dataset
    labels = np.loadtxt("labels.txt") #I copied images for balancing dataset and I produce a new labels file according to copying
    training_features = []
    for i in range(1, 267):
        string = "tr" + str(i) + ".jpg"
        image = Image.open(string)
        data = asarray(image)
        im = rgb2gray(data)
        i1, i2, i3, i4, i5, i6 = grid_features(im, 4, 8, 10)
        training_features.append((i1, i2, i3, i4, i5, i6, labels[i - 1]))

    # feature extraction from test dataset
    test_labels = np.loadtxt("test_labels.txt")
    test_features = []
    for i in range(1, 144):
        string = "ts" + str(i) + ".jpg"
        image = Image.open(string)
        data = asarray(image)
        im = rgb2gray(data)
        i1, i2, i3, i4, i5, i6 = grid_features(im, 4, 8, 10)
        test_features.append((i1, i2, i3, i4, i5, i6, test_labels[i - 1]))

    ## Normalization of Training Features
    training_feature_array = np.array(training_features)
    for i in (range(len(training_feature_array[0]) - 1)):
        mean = np.mean(training_feature_array[:, i])
        dev = np.std(training_feature_array[:, i])
        for j in range(len(training_feature_array)):
            training_feature_array[j, i] = (training_feature_array[j, i] - mean) / dev

    ## Normalization of Test Features
    test_feature_array = np.array(test_features)
    for i in (range(len(test_feature_array[0]) - 1)):
        mean = np.mean(test_feature_array[:, i])
        dev = np.std(test_feature_array[:, i])
        for j in range(len(test_feature_array)):
            test_feature_array[j, i] = (test_feature_array[j, i] - mean) / dev

    # Training
    linear_kernel = svm.SVC(C=1700, kernel="linear")
    linear_kernel.fit(training_feature_array[:, :6], training_feature_array[:, 6])
    linear_pred = linear_kernel.predict(test_feature_array[:, :6])
    rbf_kernel = svm.SVC(C=30, kernel="rbf")
    rbf_kernel.fit(training_feature_array[:, :6], training_feature_array[:, 6])
    rbf_pred = rbf_kernel.predict(test_feature_array[:, :6])

    return linear_pred, rbf_pred