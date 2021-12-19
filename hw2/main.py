import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from panorama import *

imgs_dir = "./data1"
pan_img_savename = "data1.JPG"

imgs = []
for root, dirs, files in os.walk(imgs_dir):
    files.sort()
    for f in files:
        path = os.path.join(root,f)
        imgs.append(cv.imread(path))

pan_img = imgs[0]
res_imgs = imgs[1:]

# The basic algorithm
for img in res_imgs:   
    # 'cpv' for descriptor formed by concatenated pixel values, 'sift' for SIFT descriptor 
    pan_img = panorama(img, pan_img, iters=50000, descriptor='sift')

cv.imwrite(pan_img_savename, pan_img)
# plt.imshow(pan_img[...,[2,1,0]])
# plt.show()

# Comparison of different feature descriptors
SIFT_FM = FM(imgs[-2], imgs[-1], 'sift')
CPV_FM = FM(imgs[-2], imgs[-1], 'cpv')
cv.imwrite('SIFT.JPG', SIFT_FM)
cv.imwrite('CPV.JPG', CPV_FM)
