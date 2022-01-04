import os
from glob import glob

def get_image_paths(train_path, test_path):

    train_image_paths = []
    test_image_paths = []

    train_labels = []
    test_labels = []

    for root, dirs, files in os.walk(train_path):
        for file in files:
            train_image_paths.append(os.path.join(root, file))
            train_labels.append(os.path.split(root)[-1].lower())

    for root, dirs, files in os.walk(test_path):
        for file in files:
            test_image_paths.append(os.path.join(root, file))
            test_labels.append(os.path.split(root)[-1].lower())
        
    return train_image_paths, test_image_paths, train_labels, test_labels



    




