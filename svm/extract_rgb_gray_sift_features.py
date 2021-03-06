import numpy as np
from sklearn import svm, model_selection, metrics
import os
import cv2

from .compute_kmeans_bow_features import compute_kmeans_bow_features
    
# Extracts RGB-SIFT + grayscale SIFT
def extract_rgb_gray_sift_features(images_train, images_test, **kwargs):
    print('Extracting RGB-Gray-SIFT features...')
    sift = cv2.xfeatures2d.SIFT_create()
        
    def to_sift_desc(image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (kps, desc_gray) = sift.detectAndCompute(np.array(image_gray), None)
        image_blue = [[pixel[0] for pixel in row] for row in image]
        image_green = [[pixel[1] for pixel in row] for row in image]
        image_red = [[pixel[2] for pixel in row] for row in image]
        (_, desc_blue) = sift.compute(np.array(image_blue), kps) 
        (_, desc_green) = sift.compute(np.array(image_green), kps) 
        (_, desc_red) = sift.compute(np.array(image_red), kps) 
        if desc_gray is None:
            desc_gray = []
        if desc_blue is None:
            desc_blue = []
        if desc_green is None:
            desc_green = []
        if desc_red is None:
            desc_red = []
        # Concat SIFT descriptor for all gray + 3 channels into 1 512-dim vector 
        desc = [combine_desc(gray, b, g, r) for (gray, b, g, r) in zip(desc_gray, desc_blue, desc_green, desc_red)] 
        return desc

    image_sifts_train = [to_sift_desc(image) for image in images_train]
    image_sifts_test = [to_sift_desc(image) for image in images_test]
    (X_train, X_test) = compute_kmeans_bow_features(image_sifts_train, image_sifts_test, **kwargs)

    return (X_train, X_test)

def combine_desc(gray, b, g, r):
    if gray is None:
        gray = np.zeros((128))
    if b is None:
        b = np.zeros((128))
    if g is None:
        g = np.zeros((128))
    if r is None:
        r = np.zeros((128))
    return gray + b + g + r
        