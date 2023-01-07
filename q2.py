import math
import random
import time

import cv2
import numpy as np
from scipy.linalg import null_space


def resize(img, size):

    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def normalize(img):
    # linear normalizing img between 0 to 255
    return ((img - img.min()) * (255 / (img.max() - img.min()))).astype('uint8')


def clip(img):
    # linear normalizing img between 0 to 255
    return img.clip(0, 255)


def q2(img1, img2):
    image1 = img1.copy()
    image2 = img2.copy()
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.9 * n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=4, confidence=95, maxIters=50000000)

    print(F)
    # We select only inlier points

    outpts1 = pts1[mask.ravel() == 0]
    outpts2 = pts2[mask.ravel() == 0]

    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)

    outlines1 = cv2.computeCorrespondEpilines(outpts2.reshape(-1, 1, 2), 2, F)
    outlines1 = outlines1.reshape(-1, 3)

    img3 = img1.copy()
    img4 = img2.copy()

    img3, img4 = drawlines(img3, img4, outlines1, outpts1, outpts2, 'outlier')
    img3, img4 = drawlines(img3, img4, lines1, pts1, pts2, 'inlier')

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F.T)
    lines2 = lines2.reshape(-1, 3)

    outlines2 = cv2.computeCorrespondEpilines(outpts1.reshape(-1, 1, 2), 1, F.T)
    outlines2 = outlines2.reshape(-1, 3)

    img2, img1 = drawlines(img2, img1, outlines2, outpts2, outpts1, 'outlier')
    img2, img1 = drawlines(img2, img1, lines2, pts2, pts1, 'inlier')

    cv2.imwrite('im1.jpg', img3)
    cv2.imwrite('im2.jpg', img4)

    img1 = img3
    img2 = img2

    plate = np.zeros((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype='uint8')
    plate[:img1.shape[0], :img1.shape[1]] = img1[:img1.shape[0], :img1.shape[1]]
    plate[:img2.shape[0], img2.shape[1]:] = img2[:img2.shape[0], :img2.shape[1]]
    cv2.imwrite('res05.jpg', plate)
    im1_resize = resize(image1, 0.5)
    im2_resize = resize(image2, 0.5)
    e = null_space(F)
    print(e[0] / e[2], e[1] / e[2])
    y = int((im1_resize.shape[0] + np.abs(e[0] / e[2])) / 2)
    x = int((im1_resize.shape[1] + np.abs(e[1] / e[2])) / 2)

    plate1 = np.zeros((x + 200, y + 200, 3), dtype='uint8')
    plate1[x + 100 - im1_resize.shape[0]:x + 100, y + 100 - im1_resize.shape[1]:y + 100] = im1_resize
    cv2.circle(plate1, (100, 100), 20, (0, 0, 255), -1)
    cv2.imwrite('res06.jpg', plate1)
    e_prime = null_space(F.T)
    print(e_prime[0] / e_prime[2], e_prime[1] / e_prime[2])
    y = int((im2_resize.shape[0] + np.abs(e_prime[0] / e_prime[2])) / 2)
    x = int((im2_resize.shape[1] + np.abs(e_prime[1] / e_prime[2])) / 2)

    plate1 = np.zeros((x + 200, y + 200, 3), dtype='uint8')
    plate1[x + 100 - im2_resize.shape[0]:x + 100, 100:100 + im2_resize.shape[1]] = im2_resize
    cv2.circle(plate1, (100+y, 100), 20, (0, 0, 255), -1)
    cv2.imwrite('res07.jpg', plate1)

    rand_indexes = random.sample(range(0, len(pts1)), 10)
    #print(pts1)
    rand_pts1 = np.array(pts1[rand_indexes])
    rand_pts2 = np.array(pts2[rand_indexes])

    print(rand_pts1)
    print(pts1)
    #print(rand_pts1)
    rand_lines1 = cv2.computeCorrespondEpilines(rand_pts1.reshape(-1, 1, 2), 1, F)
    rand_lines1 = rand_lines1.reshape(-1, 3)

    rand_lines2 = cv2.computeCorrespondEpilines(rand_pts2.reshape(-1, 1, 2), 1, F.T)
    rand_lines2 = rand_lines2.reshape(-1, 3)

    img8, x = drawlines(image1, image2, rand_lines1, rand_pts2, rand_pts1, 'rand')
    img7, x = drawlines(image2, image1, rand_lines2, rand_pts1, rand_pts2, 'rand')

    plate = np.zeros((max(img7.shape[0], img8.shape[0]), img7.shape[1] + img8.shape[1], 3), dtype='uint8')
    plate[:img7.shape[0], :img7.shape[1]] = img7[:img7.shape[0], :img7.shape[1]]
    plate[:img8.shape[0], img8.shape[1]:] = img8[:img8.shape[0], :img8.shape[1]]

    cv2.imwrite('res08.jpg',plate)




def drawlines(img1, img2, lines, pts1, pts2, type):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c, channels = img1.shape
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        if type == 'inlier':
            color = (0, 255, 0)
        elif  type == 'rand':
            color = (255,0,0)
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            img1 = cv2.line(img1, (x0, y0), (x1, y1), (180,180,180), 4)
        else:
            color = (0, 0, 255)

        img1 = cv2.circle(img1, tuple(pt1), 12, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 12, color, -1)
    return img1, img2


t0 = time.time()

im1 = cv2.imread('01.jpg', 1)  # load image
im2 = cv2.imread('02.jpg', 1)  # load image
q2(im1, im2)  # go to main function

t1 = time.time()
print('runtime: ' + str(int(t1 - t0)) + ' seconds')
