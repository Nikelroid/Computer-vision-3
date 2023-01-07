import math
import time
import os

import cv2
import numpy as np


def resize(size, img):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def normalize(img):
    # linear normalizing img between 0 to 255
    return ((img - img.min()) * (255 / (img.max() - img.min()))).astype('uint8')


def clip(img):
    # linear normalizing img between 0 to 255
    return img.clip(0, 255)


def q4(directory,test_directory):
    c = 0
    Train = []
    classes = []
    class_names = []
    for it in os.scandir(directory):
        if it.is_dir():
            c += 1
            Class = it.path
            if directory in Class:
                Class = Class.replace(directory, '')
            fol_path = it.path
            class_names.append(Class)

            for (root, dirs, file) in os.walk(fol_path):
                for f in file:
                    Train.append(cv2.imread(fol_path + "/" + f, 0))
                    classes.append([c])
    s = 8
    for i in range(len(Train)):
        Train[i] = resize(s, Train[i]).reshape(s ** 2)
    Train = np.array(Train, dtype=np.float32)
    classes = np.array(classes, dtype=np.float32)
    #responses = np.random.randint(0, 2, (25, 1)).astype(np.float32)

    knn = cv2.ml.KNearest_create()
    #print(Train)
    #print(classes)
    knn.train(Train, cv2.ml.ROW_SAMPLE, classes)

    #newcomer = np.random.randint(0, 100, (1, 2)).astype(np.float32)

    #print(newcomer)

    c = 0
    count = 0
    correct = 0
    prediction = []
    for it in os.scandir(test_directory):
        if it.is_dir():
            c += 1
            Class = it.path
            if test_directory in Class:
                Class = Class.replace(test_directory, '')
            fol_path = it.path
            class_names.append(Class)


            for (root, dirs, file) in os.walk(fol_path):
                for f in file:
                    count += 1
                    test = cv2.imread(fol_path + "/" + f, 0)
                    test = np.float32([resize(s, test).reshape(s ** 2)])
                    #print(test)
                    ret, r, neighbours, dist = knn.findNearest(test, 1)
                    prediction.append([r])
                    if r==c:
                        correct+=1
    print(correct,count,correct/count*100,"%")
    #print(responses)
    #print(prediction)

t0 = time.time()

directory = 'Data/Train/'
test_dir = 'Data/Test/'
q4(directory,test_dir)  # go to main function

t1 = time.time()
print('runtime: ' + str(int(t1 - t0)) + ' seconds')
