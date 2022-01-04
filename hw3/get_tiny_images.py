from PIL import Image
import numpy as np

def get_tiny_images(image_paths):
    '''
    Input : 
        image_paths: a list(N) of string where where each string is an image 
        path on the filesystem.
    Output :
        tiny image features : (N, d) matrix of resized and then vectorized tiny
        images. E.g. if the images are resized to 16x16, d would equal 256.
    '''
    
    N = len(image_paths)
    size = 16
    
    tiny_images = []
    
    for image_path in image_paths:
        image = Image.open(image_path)
        image = image.resize((size, size))
        image = (image - np.mean(image))/np.std(image)
        image = image.flatten()
        tiny_images.append(image)
        
    tiny_images = np.asarray(tiny_images)
    #print(tiny_images.shape)
    
    return tiny_images