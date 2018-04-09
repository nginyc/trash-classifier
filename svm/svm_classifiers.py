import numpy as np
import sys
import cv2
import os
import tensorflow as tf
import tensorflow_hub as hub

from .train_and_test_svm import train_and_test_svm
from common import load_images

'''
    SETTINGS (can be configured with environment variables)
'''
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 100))
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
    X = []
    sess.run(tf.global_variables_initializer())
    batches = [images[i:i + BATCH_SIZE] for i in range(0, len(images), BATCH_SIZE)]
    for batch in batches:
        bottleneck_tensors = m(batch)
        x_batch = sess.run(bottleneck_tensors)
        X.extend(x_batch)
    return X

def train_svm_raw_pixels():
    (images, image_labels) = load_images()
    X = extract_raw_pixels_feature_vectors(images)
    y = image_labels
    train_and_test_svm(X, y)

def train_svm_inception_bottleneck():
    # Potential upgrades
    # TODO: distortions, cropping, brightening the images to generate more data: \
    #   https://www.tensorflow.org/tutorials/image_retraining#bottlenecks
    # TODO: Try another SVM kernel e.g. RBF or other SVM settings: \ 
    #   https://code.oursky.com/tensorflow-svm-image-classifications-engine/
    # TODO: Find more training data?
    # TODO: Find more material types?
    # TODO: Explore misclassifications and in-depth about how bottleneck features work \
    #   (and whether other features might be better in other ways?)
    (images, image_labels) = load_images()
    X = extract_inception_bottleneck_feature_vectors(images)
    y = image_labels
    train_and_test_svm(X, y)

