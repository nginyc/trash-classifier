import tensorflow as tf

def alexnet_architecture(input_layer, params, mode):
    print("Input shape: {}".format(input_layer.shape))

    # Convolution Layer 1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=96,
        kernel_size=[11, 11],
        strides=4,
        padding="same",
        activation=tf.nn.relu)
    print("Conv1 shape: {}".format(conv1.shape))
    
    # Pooling Layer 1
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1, 
        pool_size=[3, 3], 
        strides=2)
    print("Pool1 shape: {}".format(pool1.shape))

    # Convolution Layer 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=192,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    print("Conv2 shape: {}".format(conv2.shape))
    
    # Pooling Layer 2
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2, 
        pool_size=[3, 3], 
        strides=2)
    print("Pool2 shape: {}".format(pool2.shape))
    
    # Convolution Layer 3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=288,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    print("Conv3 shape: {}".format(conv3.shape))

    # Convolution Layer 4
    conv4 = tf.layers.conv2d(
        inputs=conv3,
        filters=288,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    print("Conv4 shape: {}".format(conv4.shape))

    # Convolution Layer 5
    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=192,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    print("Conv5 shape: {}".format(conv5.shape))

    # Pooling Layer 3
    pool3 = tf.layers.max_pooling2d(
        inputs=conv5, 
        pool_size=[3, 3], 
        strides=2)
    print("Pool3 shape: {}".format(pool3.shape))

    pool3_flat = tf.reshape(pool3, [-1, pool3.shape[1] * pool3.shape[2] * pool3.shape[3]])
    print("Pool3 flattened shape: {}".format(pool3_flat.shape))

    # Dense Layer 1
    dense1 = tf.layers.dense(
        inputs=pool3_flat, 
        units=4096, 
        activation=tf.nn.relu)
    print("Dense1 shape: {}".format(dense1.shape))

    # Dropout Layer 1
    dropout1 = tf.layers.dropout(
        inputs=dense1, 
        rate=0.2, 
        training=mode == tf.estimator.ModeKeys.TRAIN)
    print("Dropout1 shape: {}".format(dropout1.shape))

    # Dense Layer 2
    dense2 = tf.layers.dense(
        inputs=dropout1, 
        units=4096, 
        activation=tf.nn.relu)
    print("Dense2 shape: {}".format(dense2.shape))

    # Dropout Layer 2
    dropout2 = tf.layers.dropout(
        inputs=dense2, 
        rate=0.2, 
        training=mode == tf.estimator.ModeKeys.TRAIN)
    print("Dropout2 shape: {}".format(dropout2.shape))
    
    # Logits Layer
    logits = tf.layers.dense(
        inputs=dropout2, 
        units=params['num_classes'])
    print("Logits shape: {}".format(logits.shape))
        
    return logits