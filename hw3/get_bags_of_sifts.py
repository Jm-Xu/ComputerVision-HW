from PIL import Image
import numpy as np
from scipy.spatial import distance
import pickle
import scipy.spatial.distance as distance
import cv2 as cv
from time import time
import pdb

def get_bags_of_sifts(image_paths):
    '''
    Input : 
        image_paths : a list(N) of training images
    Output : 
        image_feats : (N, d) feature, each row represent a feature of an image
    '''
    
    with open('vocab.pkl', 'rb') as handle:
        vocab = pickle.load(handle)
    
    image_feats = []
    
    start_time = time()
    print("Construct bags of sifts...")

    sift = cv.SIFT_create()
    for path in image_paths:
        img = cv.imread(path)
        _, descriptors = sift.detectAndCompute(img, None)
        dist = distance.cdist(vocab, descriptors, metric='euclidean')
        idx = np.argmin(dist, axis=0)
        hist, bin_edges = np.histogram(idx, bins=len(vocab))
        hist_norm = [float(i)/sum(hist) for i in hist]
        
        image_feats.append(hist_norm)
        
    image_feats = np.asarray(image_feats)
    
    end_time = time()
    print("It takes ", (start_time - end_time), " to construct bags of sifts.")
    
    return image_feats