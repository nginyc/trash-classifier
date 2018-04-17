import numpy as np
from sklearn import svm, model_selection, metrics
import os
import cv2

from .compute_kmeans_bow_features import compute_kmeans_bow_features
    
def extract_sift_features(images_gray_train, images_gray_test):
    print('Extracting SIFT features...')
    sift = cv2.xfeatures2d.SIFT_create()
    def to_sift_desc(image):
        (kps, image_sift_desc) = sift.detectAndCompute(np.array(image), None)
        if image_sift_desc is None:
            return []
        return image_sift_desc
    image_sifts_train = [to_sift_desc(image) for image in images_gray_train]
    image_sifts_test = [to_sift_desc(image) for image in images_gray_test]
    (X_train, X_test) = compute_kmeans_bow_features(image_sifts_train, image_sifts_test)

    return (X_train, X_test)