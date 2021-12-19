import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from myHomography import *
from myMatching import * 

def FM(img1, img2, descriptor='sift'):
    img1_gray = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    img2_gray = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)
    # Feature detection, descriptor
    if descriptor == 'cpv':
        # descriptor formed by concatenated pixel values
        print('using descriptor formed by concatenated pixel values')
        sift = cv.SIFT_create()
        kp1 = sift.detect(img1_gray, None)
        des1 = np.array([cv.getRectSubPix(img1_gray, (11, 11), kp.pt).reshape(-1) for kp in kp1])
        kp2 = sift.detect(img2_gray, None)
        des2 = np.array([cv.getRectSubPix(img2_gray, (11, 11), kp.pt).reshape(-1) for kp in kp2])

    else:
        # SIFT descriptor
        print('using SIFT descriptor')
        sift = cv.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1_gray, None)
        kp2, des2 = sift.detectAndCompute(img2_gray, None)

    # Feature matching with ratio test
    # bf = cv.BFMatcher()
    # matches = bf.knnMatch(des1,des2,k=2)
    # my function for Feature matching
    matches = myknn(des1, des2, 2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    FMimg = cv.drawMatches(img1,kp1,img2,kp2,good,None)
    return FMimg

def panorama(img1, img2, iters, descriptor='sift'):
    img1_gray = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    img2_gray = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)
    # Feature detection, descriptor
    if descriptor == 'cpv':
        # descriptor formed by concatenated pixel values
        print('using descriptor formed by concatenated pixel values')
        sift = cv.SIFT_create()
        kp1 = sift.detect(img1_gray, None)
        des1 = np.array([cv.getRectSubPix(img1_gray, (11, 11), kp.pt).reshape(-1) for kp in kp1])
        kp2 = sift.detect(img2_gray, None)
        des2 = np.array([cv.getRectSubPix(img2_gray, (11, 11), kp.pt).reshape(-1) for kp in kp2])

    else:
        # SIFT descriptor
        print('using SIFT descriptor')
        sift = cv.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1_gray, None)
        kp2, des2 = sift.detectAndCompute(img2_gray, None)

    # Feature matching with ratio test
    # bf = cv.BFMatcher()
    # matches = bf.knnMatch(des1,des2,k=2)
    # my function for Feature matching
    matches = myknn(des1, des2, 2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    # Estimation of homography
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    # opencv function for Estimation of homography
    # H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0, maxIters=2000)
    # my function for Estimation of homography
    H = myHomography(src_pts, dst_pts, 5.0, maxIters=50)

    # Image stitching
    h,w = img1_gray.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,H)
    c1, c2, c3, c4 = dst.reshape(4,-1)
    botrightw = max(c1[0], c2[0], c3[0], c4[0])
    botrighth = max(c1[1], c2[1], c3[1], c4[1])
    topleftw  = min(c1[0], c2[0], c3[0], c4[0])
    toplefth  = min(c1[1], c2[1], c3[1], c4[1])
    dh = int(min(0, toplefth))
    dw = int(min(0, topleftw))
    tw  = int(max(img2_gray.shape[1], botrightw) - dw)
    th  = int(max(img2_gray.shape[0], botrighth) - dh)

    affm = np.float32([[1, 0, -dw], [0, 1, -dh]])
    affm2 = np.float32([[1, 0, -dw], [0, 1, -dh], [0, 0, 1]])
    aff = cv.warpAffine(img2, affm, (tw, th))
    dst = cv.warpPerspective(img1, affm2.dot(H), (tw, th))

    # matrix operation for accelerating linear blending to decide overlapped regions
    zero_idx = np.where(dst == 0)
    dst[zero_idx] += aff[zero_idx]

    # linear blending to decide overlapped regions
    # res = np.zeros_like(aff) 
    # for i in range(aff.shape[0]):
    #     for j in range(aff.shape[1]):
    #         if aff[i, j] == 0 or dst[i ,j] == 0:
    #             res[i, j] = aff[i, j] + dst[i, j]
    #         else:
    #             res[i, j] = aff[i, j]
    # res = [aff[i, j] + dst[i, j] if aff[i, j] == 0 or dst[i ,j] == 0 else (aff[i, j] + dst[i, j]) / 2 for j in range(aff.shape[1]) for i in range(aff.shape[0])]

    # plt.imshow(dst, 'gray')
    # plt.show()

    return dst