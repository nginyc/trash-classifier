import numpy as np
from sklearn import svm, model_selection, metrics
import os
import cv2

from .compute_kmeans_bow_features import compute_kmeans_bow_features
    
def extract_orb_features(images_gray_train, images_gray_test):
    print('Extracting ORB features...')
    orb = cv2.ORB_create()
    def to_orb_desc(image):
        (kps, image_orb_desc) = orb.detectAndCompute(np.array(image), None)
        if image_orb_desc is None:
            return []
        return image_orb_desc
    image_orbs_train = [to_orb_desc(image) for image in images_gray_train]
    image_orbs_test = [to_orb_desc(image) for image in images_gray_test]
    (X_train, X_test) = compute_kmeans_bow_features(image_orbs_train, image_orbs_test)
    return (X_train, X_test)
