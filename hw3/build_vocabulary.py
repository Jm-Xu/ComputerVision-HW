from PIL import Image
import numpy as np
import cv2 as cv
from time import time
from sklearn.cluster import KMeans


# This function will sample SIFT descriptors from the training images,
# cluster them with kmeans, and then return the cluster centers.

def build_vocabulary(image_paths, vocab_size):
    '''
    Input : 
        image_paths : a list of training image path
        vocal size : number of clusters desired
    Output :
        Clusters centers of Kmeans
    '''

    bag_of_features = []
    
    print("Extract SIFT features")
    
    sift = cv.SIFT_create()
    for path in image_paths:
        # if np.random.randint(0,10) > 1:
        #     continue
        img = cv.imread(path)
        _, descriptors = sift.detectAndCompute(img, None)
        bag_of_features.append(descriptors)
    bag_of_features = np.concatenate(bag_of_features, axis=0).astype('float32')
    
    print("Compute vocab")
    start_time = time()
    kmeans = KMeans(n_clusters=vocab_size, random_state=0).fit(bag_of_features)
    vocab = kmeans.cluster_centers_
    # _, _, vocab = cv.kmeans(bag_of_features, vocab_size, None)        
    end_time = time()
    print("It takes ", (start_time - end_time), " to compute vocab.")
    
    return vocab