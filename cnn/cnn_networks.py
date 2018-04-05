from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

def alexnet_layers_fn(input_layer, mode):

    print("Input shape: ", input_layer.shape)

    # Convolution Layer 1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=96,
        kernel_size=[11, 11],
        strides=4,
        padding="same",
        activation=tf.nn.relu)

    print("Conv1 shape: ", conv1.shape)
    
    # Pooling Layer 1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1, 
        pool_size=[3, 3], 
        strides=2)

    print("Pool1 shape: ", pool1.shape)

    # Convolution Layer 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=192,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    print("Conv2 shape: ", conv2.shape)
    
    # Pooling Layer 2
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2, 
        pool_size=[3, 3], 
        strides=2)

    print("Pool2 shape: ", pool2.shape)
    
    # Convolution Layer 3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=288,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    print("Conv3 shape: ", conv3.shape)

    # Convolution Layer 4
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=288,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    print("Conv4 shape: ", conv4.shape)

    # Convolution Layer 5
    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=192,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)

    print("Conv5 shape: ", conv5.shape)

    # Pooling Layer 3
    pool3 = tf.layers.max_pooling2d(
        inputs=conv5, 
        pool_size=[3, 3], 
        strides=2)

    print("Pool3 shape: ", pool3.shape)

    pool3_flat = tf.reshape(pool3, [-1, pool3.shape[1] * pool3.shape[2] * pool3.shape[3]])

    print("Pool3 flat shape: ", pool3_flat.shape)

    # Dense Layer 1
    dense1 = tf.layers.dense(
        inputs=pool3_flat, 
        units=4096, 
        activation=tf.nn.relu)

    print("Dense1 shape: ", dense1.shape)

    # Dropout Layer 1
    dropout1 = tf.layers.dropout(
        inputs=dense1, 
        rate=0.2, 
        training=mode == tf.estimator.ModeKeys.TRAIN)

    print("Dropout1 shape: ", dropout1.shape)

    # Dense Layer 2
    dense2 = tf.layers.dense(
        inputs=dropout1, 
        units=4096, 
        activation=tf.nn.relu)

    print("Dense2 shape: ", dense2.shape)

    # Dropout Layer 2
    dropout2 = tf.layers.dropout(
        inputs=dense2, 
        rate=0.2, 
        training=mode == tf.estimator.ModeKeys.TRAIN)

    print("Dropout2 shape: ", dropout2.shape)
    
    # Logits Layer
    logits = tf.layers.dense(
        inputs=dropout2, 
        units=6)
        
    print("Logits shape: ", logits.shape)
    return logits