import numpy as np
import cv2 as cv

def myHomography(src_pts, dst_pts, thres, maxIters=2000):
    inlier_num = 0
    inlier_idxs = []
    for i in range(maxIters):
        num = 0
        idxs = np.random.randint(0, src_pts.shape[0], 4)
        src, dst = src_pts[idxs, ...], dst_pts[idxs, ...] 
        H = dlt(src, dst)
        for pt in range(src_pts.shape[0]):
            if np.linalg.norm(dst_pts[pt][None,...] - cv.perspectiveTransform(src_pts[pt][None,...],H)) < thres:
                num += 1
        if num > inlier_num:
            inlier_num = num
            inlier_idxs = idxs
    
    H = mestimator(inlier_idxs, src_pts, dst_pts)

    return H

def mestimator(inlier_idxs, src_pts, dst_pts):
    src, dst = src_pts[inlier_idxs, ...], dst_pts[inlier_idxs, ...]
    A = []
    for i in range(src.shape[0]):
        eq1 = [0, 0, 0, -src[i,0,0], -src[i,0,1], -1, dst[i,0,1]*src[i,0,0], dst[i,0,1]*src[i,0,1], dst[i,0,1]]
        eq2 = [src[i,0,0], src[i,0,1], 1, 0, 0, 0, -dst[i,0,0]*src[i,0,0], -dst[i,0,0]*src[i,0,1], -dst[i,0,0]]
        A.append(eq1)
        A.append(eq2)
    u, s, vh = np.linalg.svd(np.array(A))
    H = (vh[-1, :] / vh[-1, -1]).reshape(3, 3)
    return H

def dlt(src, dst):
    A = []
    for i in range(src.shape[0]):
        eq1 = [0, 0, 0, -src[i,0,0], -src[i,0,1], -1, dst[i,0,1]*src[i,0,0], dst[i,0,1]*src[i,0,1], dst[i,0,1]]
        eq2 = [src[i,0,0], src[i,0,1], 1, 0, 0, 0, -dst[i,0,0]*src[i,0,0], -dst[i,0,0]*src[i,0,1], -dst[i,0,0]]
        A.append(eq1)
        A.append(eq2)
    u, s, vh = np.linalg.svd(np.array(A))
    H = (vh[-1, :] / vh[-1, -1]).reshape(3, 3)
    return H
