import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def panorama(img1, img2, iters):
    # Feature detection, descriptor
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Feature matching with ratio test
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    # Estimation of homography
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0, maxIters=2000)

    # Image stitching
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,H)
    c1, c2, c3, c4 = dst.reshape(4,-1)
    botrightw = max(c1[0], c2[0], c3[0], c4[0])
    botrighth = max(c1[1], c2[1], c3[1], c4[1])
    topleftw  = min(c1[0], c2[0], c3[0], c4[0])
    toplefth  = min(c1[1], c2[1], c3[1], c4[1])
    dh = int(min(0, toplefth))
    dw = int(min(0, topleftw))
    tw  = int(max(img2.shape[1], botrightw) - dw)
    th  = int(max(img2.shape[0], botrighth) - dh)

    affm = np.float32([[1, 0, -dw], [0, 1, -dh]])
    affm2 = np.float32([[1, 0, -dw], [0, 1, -dh], [0, 0, 1]])
    aff = cv.warpAffine(img2, affm, (tw, th))
    dst = cv.warpPerspective(img1, affm2.dot(H), (tw, th))

    # res = np.zeros_like(aff)
    zero_idx = np.where(dst == 0)
    dst[zero_idx] += aff[zero_idx] 
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