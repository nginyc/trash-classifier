from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def alex_net(input_layer):

    # Convolution Layer 1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=96,
        kernel_size=[11, 11],
        padding="same",
        activation=tf.nn.relu)
    
    # Pooling Layer 1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1, 
        pool_size=[3, 3], 
        strides=2)

    # Convolution Layer 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=192,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    
    # Pooling Layer 2
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2, 
        pool_size=[3, 3], 
        strides=2)
    
    # Convolution Layer 3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=288,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Convolution Layer 4
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=288,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Convolution Layer 5
    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=192,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer 3
    pool3 = tf.layers.max_pooling2d(
        inputs=conv5, 
        pool_size=[3, 3], 
        strides=2)

    pool3_flat = tf.reshape(pool2, [-1, 7 * 7 * 192])

    # Dense Layer 1
    dense1 = tf.layers.dense(
        inputs=pool3_flat, 
        units=4096, 
        activation=tf.nn.relu)

    # Dropout Layer 1
    dropout1 = tf.layers.dropout(
        inputs=dense1, 
        rate=0.2, 
        training=mode == tf.estimator.ModeKeys.TRAIN)

    # Dense Layer 2
    dense2 = tf.layers.dense(
        inputs=dropout1, 
        units=4096, 
        activation=tf.nn.relu)

    # Dropout Layer 2
    dropout2 = tf.layers.dropout(
        inputs=dense2, 
        rate=0.2, 
        training=mode == tf.estimator.ModeKeys.TRAIN)
    
    # Logits Layer
    logits = tf.layers.dense(
        inputs=dropout2, 
        units=6)

    return logits