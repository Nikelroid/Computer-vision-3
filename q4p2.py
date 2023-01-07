import math
import time
import os
import cv2
import numpy as np


def q4(directory,test_directory):

    clusters = 85
    sift = cv2.SIFT_create(550)
    descriptors = []
    c = 0
    counter = 0
    Train = []
    classes = []
    class_names = []
    descriptors_count = []
    print('phase 1 started')
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
                    counter += 1
                    im =cv2.imread(fol_path + "/" + f, 0)
                    keypoint, descriptor = sift.detectAndCompute(im,None)
                    for i in range (len(descriptor)):
                        descriptors.append(descriptor[i])
                    classes.append(c)
                    descriptors_count.append(len(descriptor))

            print(c,'/15 - 1')

    train_count = counter
    print(train_count)
    prob = [0 for m in range (clusters)]
    descriptors = np.array(descriptors)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(descriptors, clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)


    c1 = 0

    for i in range (counter):
        checked = []
        Train.append([0 for m in range (clusters)])
        for j in range (descriptors_count[i]):
            Train[-1][label[c1][0]] += 1
            if (not(label[c1][0] in checked)):
                prob[label[c1][0]] += 1
            checked.append(label[c1][0])
            c1+=1

    Train = np.array(Train, dtype=np.float32)
    classes = np.array(classes, dtype='int')

    for q in range(len(prob)):
        print(prob[q],(train_count/prob[q])-0.1)
        prob[q] = ((train_count/prob[q])-0.1)

    for train in Train:
        for i in range(len(train)):
            train[i] *= prob[i]

    knn = cv2.ml.KNearest_create()
    knn.train(Train, cv2.ml.ROW_SAMPLE, classes)

    c = 0
    count = 0
    correct = 0
    prediction = []
    print('phase 2 started')
    for it in os.scandir(test_directory):
        if it.is_dir():

            Class = it.path
            if test_directory in Class:
                Class = Class.replace(test_directory, '')
            fol_path = it.path
            # class_names.append(Class)
            prediction = []
            count = 0
            for (root, dirs, file) in os.walk(fol_path):
                for f in file:
                    count += 1
                    test = cv2.imread(fol_path + "/" + f, 0)
                    keypoint, descriptor = sift.detectAndCompute(test, None)

                    t = []
                    for des in descriptor:
                        distances = np.array([np.linalg.norm(center[q] - des) for q in range(clusters)])
                        t.append(distances.argmin())
                    t = np.array(t)
                    test_histogram = [0 for m in range(clusters)]
                    for j in range(len(t)):
                        test_histogram[int(t[j])] += 1
                    test_histogram = np.array([test_histogram], dtype=np.float32)
                    for i in range(len(test_histogram)):
                        test_histogram[i] *= prob[i]
                    r = clf.predict(test_histogram)
                    prediction.append(r)
                    if r == c:
                        correct += 1
            for u in range(15):
                confusion[u][c] = prediction.count(u)
            c += 1
            print(c, '/15 - 2')

    print(correct, count, correct / count * 100, "%")
    # print(responses)
    # print(prediction)

    print(confusion)

    df_cm = pd.DataFrame(confusion, class_names, class_names)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.show()


t0 = time.time()
directory = 'Data/Train/'
test_dir = 'Data/Test/'
q4(directory,test_dir)  # go to main function

t1 = time.time()
print('runtime: ' + str(int(t1 - t0)) + ' seconds')
