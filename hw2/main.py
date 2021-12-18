import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from panorama import *

imgs_dir = "./data1"

imgs_gray = []
for root, dirs, files in os.walk(imgs_dir):
    for f in files:
        path = os.path.join(root,f)
        imgs_gray.append(cv.imread(path, cv.IMREAD_GRAYSCALE))

pan_img = imgs_gray[0]
res_imgs = imgs_gray[1:]

for img in res_imgs:   
    # 'cpv' for descriptor formed by concatenated pixel values, 'sift' for SIFT descriptor 
    pan_img = panorama(img, pan_img, iters=50000, descriptor='sift')

plt.imshow(pan_img, 'gray')
plt.show()
