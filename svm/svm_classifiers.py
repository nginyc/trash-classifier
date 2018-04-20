import numpy as np
import sys
import cv2
import os
from sklearn import svm, metrics, model_selection
from scipy import stats
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from common import load_images
from .extract_inception_bottleneck_features import extract_inception_bottleneck_features
from .extract_sift_features import extract_sift_features
from .extract_orb_features import extract_orb_features
from .extract_rgb_sift_features import extract_rgb_sift_features
from .extract_rgb_gray_sift_features import extract_rgb_gray_sift_features

'''
    SETTINGS (can be configured with environment variables)
'''
SVM_C_PARAM = float(os.environ.get('SVM_C_PARAM', 1))
SVM_GAMMA_PARAM = float(os.environ.get('SVM_GAMMA_PARAM', 0))
MAX_ITERATIONS = int(os.environ.get('MAX_ITERATIONS', 10000))
IF_VISUALIZE_FEATURES = bool(os.environ.get('IF_VISUALIZE_FEATURES', False))
TEST_SET_RATIO = float(os.environ.get('TEST_SET_RATIO', 0.3))

'''
    Trains and tests an SVM given a method that extracts features from images
'''
def train(extract_features, if_grayscale=False, if_normalize_images=True,
    svm_c_param=SVM_C_PARAM, svm_gamma_param=SVM_GAMMA_PARAM, **kwargs):
    # Load images
    (images, labels) = load_images(if_grayscale=if_grayscale, if_normalize_images=if_normalize_images)

    # Split images into train and test set
    test_set_ratio = TEST_SET_RATIO
    print('Splitting dataset to train & test set with test_set_ratio=' + str(test_set_ratio) + '...')
    (images_train, images_test, labels_train, labels_test) = \
        model_selection.train_test_split(images, labels, test_size=test_set_ratio)

    # Extract features from images
    print('Extracting features for train & test set...')
    (X_train, X_test) = extract_features(images_train, images_test, **kwargs)
    y_train = labels_train
    y_test = labels_test

    # Visualize features
    if IF_VISUALIZE_FEATURES:
        from .visualize_features import visualize_features
        visualize_features(X_train, y_train)

    # Train SVM model with train set
    gamma = svm_gamma_param if svm_gamma_param != 0 else 'auto'
    print('Training SVM with c=' + str(svm_c_param) + ', gamma=' + str(svm_gamma_param) + '...')
    model = svm.SVC(max_iter=MAX_ITERATIONS, C=svm_c_param, gamma=gamma, probability=True)
    model.fit(X_train, y_train)
    # Return the distance of the sample from all other hyperplane
    # Higher value of the distance higher confidence
    confidence_of_model = model.predict_proba(X_train)
    confidence_example = np.amax(confidence_of_model, axis=1)
    print(stats.describe(confidence_example))
    #print(str(confidence_average))
    #print(str(confidence_of_model))


    # Test SVM model
    y_train_predict = np.array(model.predict(X_train))
    y_test_predict = np.array(model.predict(X_test))
    (train_accuracy, train_confusion_matrix) = get_accuracy(y_train_predict, y_train)
    (test_accuracy, test_confusion_matrix) = get_accuracy(y_test_predict, y_test)
    print('Train accuracy: ' + str(train_accuracy))
    print('Train confusion matrix:')
    print(train_confusion_matrix)
    print('Test accuracy: ' + str(test_accuracy))
    print('Test confusion matrix:')
    print(test_confusion_matrix)
    return test_accuracy
    
def get_accuracy(y_predict, y):
    accuracy = np.sum(y_predict == y).__float__() / float(len(y))
    confusion_matrix = metrics.confusion_matrix(y_predict, y)
    return (accuracy, confusion_matrix)

def train_svm_inception_bottleneck():
    return train(extract_inception_bottleneck_features)

def train_svm_sift_kmeans(**kwargs):
    return train(extract_sift_features, if_grayscale=True, **kwargs)

def train_svm_orb_kmeans(**kwargs):
    return train(extract_orb_features, if_grayscale=True, **kwargs)

def train_svm_rgb_sift_kmeans(**kwargs):
    return train(extract_rgb_sift_features, **kwargs)

def train_svm_rgb_gray_sift_kmeans(**kwargs):
    return train(extract_rgb_gray_sift_features, **kwargs)

def train_skopt_svm_sift_kmeans_params():
    space = [
        Categorical(['svm_sift_kmeans', 'svm_rgb_sift_kmeans', 'svm_rgb_gray_sift_kmeans'], 
            name='mode', prior=[0.25, 0.25, 0.5]),
        Categorical([0.00001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000], 
             name='svm_c_param', prior=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.55, 0.05, 0.05]),
        Categorical([0.00001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000], 
             name='svm_gamma_param', prior=[0.05, 0.55, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]),
        Categorical([True, False], name='if_normalize_images', prior=[0.9, 0.1]),
        Categorical([None, 'sqrt_keypoints', 'instances_count', 'average_keypoints'], 
            name='kmeans_clusters', prior=[0.2, 0.4, 0.2, 0.2]),
        Categorical(['l1', 'min_max', 'binary'], name='kmeans_bow_features_normalization', prior=[0.5, 0.25, 0.25])
    ]

    @use_named_args(space)
    def min_func(
        mode, svm_c_param, svm_gamma_param, if_normalize_images, 
        kmeans_clusters, kmeans_bow_features_normalization
    ):
        print('Using parameters:', mode, svm_c_param, svm_gamma_param, if_normalize_images, 
            kmeans_clusters, kmeans_bow_features_normalization)

        train_method = ({
            'svm_sift_kmeans': train_svm_sift_kmeans,
            'svm_rgb_sift_kmeans': train_svm_rgb_sift_kmeans,
            'svm_rgb_gray_sift_kmeans': train_svm_rgb_gray_sift_kmeans,
        })[mode]
        
        accuracy = train_method(svm_c_param=svm_c_param, svm_gamma_param=svm_gamma_param,
            kmeans_clusters=kmeans_clusters, 
            kmeans_bow_features_normalization=kmeans_bow_features_normalization,
            if_normalize_images=if_normalize_images)
        return -accuracy # Higher accuracy => smaller value
        
    res_gp = gp_minimize(min_func, space, n_calls=20, verbose=True)

    print('Best parameters:', mode, svm_c_param, svm_gamma_param, if_normalize_images, 
            kmeans_clusters, kmeans_bow_features_normalization)
