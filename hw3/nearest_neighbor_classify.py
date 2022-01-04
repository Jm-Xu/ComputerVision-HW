from __future__ import print_function

import numpy as np
import scipy.spatial.distance as distance

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, CATEGORIES):
    '''
    Input : 
        train_image_feats : 
            image_feats is an (N, d) matrix, where d is the 
            dimensionality of the feature representation.
        train_labels : 
            image_feats is a list of string, each string
            indicate the ground truth category for each training image. 
        test_image_feats : 
            image_feats is an (M, d) matrix, where d is the 
            dimensionality of the feature representation.
    Output :
        test_predicts : 
            a list(M) of string, each string indicate the predict
            category for each testing image.
    '''
    
    K = 1
    
    N = train_image_feats.shape[0]
    M = test_image_feats.shape[0]
    d = train_image_feats.shape[1] # d are same in both train and test
    
    dist = distance.cdist(test_image_feats, train_image_feats, metric='euclidean')

    test_predicts = []
    
    for each in dist:
        label = []
        idx = np.argsort(each)
        for i in range(K):
            label.append(train_labels[idx[i]])
        
        #print(label)
        amount = 0
        for item in CATEGORIES:
            if label.count(item) > amount:
                label_final = item
    
        test_predicts.append(label_final)
        
    
    return test_predicts