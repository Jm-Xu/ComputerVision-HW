from __future__ import print_function
from random import shuffle
import os
import pickle

from get_image_paths import get_image_paths
from get_tiny_images import get_tiny_images
from build_vocabulary import build_vocabulary
from get_bags_of_sifts import get_bags_of_sifts
from visualize import visualize

from nearest_neighbor_classify import nearest_neighbor_classify
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Step 0: Set up parameters, category list, and image paths.

train_path = './train'
test_path = './test'

#This is the list of categories / directories to use. The categories are
#somewhat sorted by similarity so that the confusion matrix looks more
#structured (indoor and then urban and then rural).

CATEGORIES_ORI = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
              'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
              'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']
CATEGORIES = [i.lower() for i in CATEGORIES_ORI]

CATE2ID = {v: k for k, v in enumerate(CATEGORIES)}

ABBR_CATEGORIES = ['Kit', 'Sto', 'Bed', 'Liv', 'Off', 'Ind', 'Sub',
                   'Cty', 'Bld', 'St', 'HW', 'OC', 'Cst', 'Mnt', 'For']

# Tiny images representation or Bag of SIFT representation
# FEATURE = 'tiny_image'
FEATURE  = 'bag_of_sift'


def main():
    #This function returns arrays containing the file path for each train
    #and test image, as well as arrays with the label of each train and test image. 
    print("Getting paths and labels for all train and test data")
    train_image_paths, test_image_paths, train_labels, test_labels = \
        get_image_paths(train_path, test_path)

    # Step 1:
    # Represent each image with the appropriate feature
    # Each function to construct features should return an N x d matrix, where
    # N is the number of paths passed to the function and d is the 
    # dimensionality of each image representation. See the starter code for
    # each function for more details.

    if FEATURE == 'tiny_image':
        # get_tiny_images
        train_image_feats = get_tiny_images(train_image_paths)
        test_image_feats = get_tiny_images(test_image_paths)

    elif FEATURE == 'bag_of_sift':
        # build_vocabulary
        if os.path.isfile('vocab.pkl') is False:
            print('No existing visual word vocabulary found. Computing one from training images\n')
            vocab_size = 50   ### Vocab_size is up to you. Larger values will work better (to a point) but be slower to comput.
            vocab = build_vocabulary(train_image_paths, vocab_size)
            with open('vocab.pkl', 'wb') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if os.path.isfile('train_image_feats_1.pkl') is False:
            # get_bags_of_sifts.py
            train_image_feats = get_bags_of_sifts(train_image_paths);
            with open('train_image_feats_1.pkl', 'wb') as handle:
                pickle.dump(train_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('train_image_feats_1.pkl', 'rb') as handle:
                train_image_feats = pickle.load(handle)

        if os.path.isfile('test_image_feats_1.pkl') is False:
            test_image_feats  = get_bags_of_sifts(test_image_paths);
            with open('test_image_feats_1.pkl', 'wb') as handle:
                pickle.dump(test_image_feats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('test_image_feats_1.pkl', 'rb') as handle:
                test_image_feats = pickle.load(handle)
    else:
        raise NameError('Unknown feature type')

    # Step 2: 
    # Classify each test image by training and using the appropriate classifier
    # Each function to classify test features will return an N x 1 array,
    # where N is the number of test cases and each entry is a string indicating
    # the predicted category for each test image. Each entry in
    # 'predicted_categories' must be one of the 15 strings in 'categories',
    # 'train_labels', and 'test_labels.

    # nearest_neighbor_classify
    predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, CATEGORIES)


    accuracy = float(len([x for x in zip(test_labels,predicted_categories) if x[0]==x[1]]))/float(len(test_labels))
    print("Accuracy = ", accuracy)
    
    for category in CATEGORIES:
        accuracy_each = float(len([x for x in zip(test_labels,predicted_categories) if x[0]==x[1] and x[0]==category]))/float(test_labels.count(category))
        print(str(category) + ': ' + str(accuracy_each))
    
    test_labels_ids = [CATE2ID[x] for x in test_labels]
    predicted_categories_ids = [CATE2ID[x] for x in predicted_categories]
    train_labels_ids = [CATE2ID[x] for x in train_labels]
    
    # Step 3: Build a confusion matrix, score the recognition system and visualize
   
    build_confusion_mtx(test_labels_ids, predicted_categories_ids, ABBR_CATEGORIES)
    visualize(CATEGORIES, test_image_paths, test_labels_ids, predicted_categories_ids, train_image_paths, train_labels_ids)

def build_confusion_mtx(test_labels_ids, predicted_categories, abbr_categories):
    # Compute confusion matrix
    cm = confusion_matrix(test_labels_ids, predicted_categories)
    np.set_printoptions(precision=2)
    '''
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm, CATEGORIES)
    '''
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #print('Normalized confusion matrix')
    #print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, abbr_categories, title='Normalized confusion matrix')

    plt.show()
     
def plot_confusion_matrix(cm, category, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(category))
    plt.xticks(tick_marks, category, rotation=45)
    plt.yticks(tick_marks, category)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == '__main__':
    main()
