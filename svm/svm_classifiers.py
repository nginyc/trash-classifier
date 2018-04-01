import numpy as np
import sys
import cv2
import matplotlib
from sklearn import svm, model_selection, metrics
matplotlib.use('TkAgg')  # Fixs matplotlib issue in virtualenv
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import tensorflow_hub as hub

from common import load_images

'''
    SETTINGS (can be configured with environment variables)
'''
MAX_ITERATIONS = int(os.environ.get('MAX_ITERATIONS', 1000))
KFOLD_SPLITS = int(os.environ.get('KFOLD_SPLITS', 5))
KFOLD_RANDOM_STATE = int(os.environ.get('KFOLD_RANDOM_STATE', 666))
TFHUB_INCEPTION_V3_MODULE_SPEC_URL = os.environ.get('TFHUB_INCEPTION_V3_MODULE_SPEC_URL', 
    'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1')

def extract_raw_pixels_feature_vectors(images):
    print('Extracting raw pixels as feature vectors...')
    X = [image.flatten() for image in images]
    return X

def extract_inception_bottleneck_feature_vectors(images):
    print('Downloading Inception V3 Tensorflow Hub model spec...')
    module_spec = hub.load_module_spec(TFHUB_INCEPTION_V3_MODULE_SPEC_URL)
    print('Extracting inception bottleneck feature vectors...')
    module = hub.Module(module_spec)
    (image_height, image_width) = hub.get_expected_image_size(module)
    images = [tf.image.convert_image_dtype(x, tf.float32) for x in images]
    images = [tf.image.resize_images(x, (image_height, image_width)) for x in images]
    sess = tf.Session()
    m = hub.Module(module_spec)
    bottleneck_tensors = m(images)
    sess.run(tf.global_variables_initializer())
    X = sess.run(bottleneck_tensors)
    return X

def train_and_test_svm(X, y):
    print('Training & testing model...')
    X = np.array(X)
    y = np.array(y)
    
    kf = model_selection.KFold(
        n_splits=KFOLD_SPLITS, random_state=KFOLD_RANDOM_STATE, shuffle=True
    )

    y_test_predict_list = []
    y_test_list = []
    for (i, (train_indices, test_indices)) in enumerate(kf.split(X)):
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
        print('Training model for fold ' + str(i) + '...')
        model = svm.SVC(max_iter=MAX_ITERATIONS)
        model.fit(X_train, y_train)
        print('\n')
        print('Testing model for fold ' + str(i) + '...')
        y_test_predict = np.array(model.predict(X_test))
        accuracy = np.sum(y_test_predict == y_test) / y_test.size
        confusion_matrix = metrics.confusion_matrix(y_test_predict, y_test)
        print('\n')
        print('Accuracy: ' + str(accuracy))
        print('Confusion matrix:')
        print(confusion_matrix)
        y_test_predict_list.append(y_test_predict)
        y_test_list.append(y_test)

    y_test_predict_comb = np.array(y_test_predict_list).flatten()
    y_test_comb = np.array(y_test_list).flatten()
    accuracy_comb = np.sum(y_test_predict_comb == y_test_comb) / y_test_comb.size
    confusion_matrix_comb = metrics.confusion_matrix(y_test_predict_comb, y_test_comb)
    print('\n\n')
    print('Combined accuracy: ' + str(accuracy_comb))
    print('Combined confusion matrix:')
    print(confusion_matrix_comb)

def train_svm_raw_pixels():
    (images, image_labels) = load_images()
    X = extract_raw_pixels_feature_vectors(images)
    y = image_labels
    train_and_test_svm(X, y)

def train_svm_inception_bottleneck():
    (images, image_labels) = load_images()
    X = extract_inception_bottleneck_feature_vectors(images)
    y = image_labels
    train_and_test_svm(X, y)

