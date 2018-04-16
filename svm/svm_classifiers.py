import numpy as np
import sys
import cv2
import os
from sklearn import svm, metrics, model_selection

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
def train(extract_features, if_grayscale=False):
    # Load images
    (images, labels) = load_images(if_grayscale=if_grayscale)

    # Split images into train and test set
    test_set_ratio = TEST_SET_RATIO
    print('Splitting dataset to train & test set with test_set_ratio=' + str(test_set_ratio) + '...')
    (images_train, images_test, labels_train, labels_test) = \
        model_selection.train_test_split(images, labels, test_size=test_set_ratio)

    # Extract features from images
    print('Extracting features for train & test set...')
    (X_train, X_test) = extract_features(images_train, images_test)
    y_train = labels_train
    y_test = labels_test

    # Visualize features
    if IF_VISUALIZE_FEATURES:
        from .visualize_features import visualize_features
        visualize_features(X_train, y_train)

    # Train SVM model with train set
    gamma = SVM_GAMMA_PARAM if SVM_GAMMA_PARAM != 0 else 'auto'
    model = svm.SVC(max_iter=MAX_ITERATIONS, C=SVM_C_PARAM, gamma=gamma, probability=True)
    model.fit(X_train, y_train)
    # Return the distance of the sample from all other hyperplane
    # Higher value of the distance higher confidence
    confidence_of_model = model.predict_proba(X_train)
    confidence_average = np.amax(confidence_of_model, axis=1)
    print(str(confidence_average))
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
    
def get_accuracy(y_predict, y):
    accuracy = np.sum(y_predict == y).__float__() / float(len(y))
    confusion_matrix = metrics.confusion_matrix(y_predict, y)
    return (accuracy, confusion_matrix)

def train_svm_inception_bottleneck():
    train(extract_inception_bottleneck_features)

def train_svm_sift_kmeans():
    train(extract_sift_features, if_grayscale=True)

def train_svm_orb_kmeans():
    train(extract_orb_features, if_grayscale=True)

def train_svm_rgb_sift_kmeans():
    train(extract_rgb_sift_features)

def train_svm_rgb_gray_sift_kmeans():
    train(extract_rgb_gray_sift_features)
