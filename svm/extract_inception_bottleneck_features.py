import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 100))
TFHUB_INCEPTION_V3_MODULE_SPEC_URL = os.environ.get('TFHUB_INCEPTION_V3_MODULE_SPEC_URL', 
    'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1')
    
def extract_inception_bottleneck_features(images_train, images_test):
    # Combine train & test set to single set of images
    images = images_train + images_test
    
    print('Extracting inception bottleneck features...')
    module_spec = hub.load_module_spec(TFHUB_INCEPTION_V3_MODULE_SPEC_URL)
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

    # Recover train & test set 
    X_train = X[:len(images_train)]
    X_test = X[len(images_train):]
    return (X_train, X_test)