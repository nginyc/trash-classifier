import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
# from PIL import  Image

DATASET_PATH =  os.environ.get('DATASET_PATH',
    os.path.dirname(__file__) + '/../data/garythung-trashnet')

data_path = [os.path.join(DATASET_PATH, o, 'train.tfrecords') for o in os.listdir(DATASET_PATH)
                                if os.path.isdir(os.path.join(DATASET_PATH, o))]

def run():
    return_data = np.empty((0))
    return_label = np.empty((0))

    with tf.Session() as sess:
        print("start")
        feature = {'image': tf.FixedLenFeature([], tf.string),
                   'label': tf.FixedLenFeature([], tf.int64)}
        # Create a list of filenames and pass it to a queue
        # print(data_path)
        filename_queue = tf.train.string_input_producer(data_path, num_epochs=1)
        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)
        # Convert the image data from string back to the numbers
        image = tf.decode_raw(features['image'], tf.uint8)
        image = tf.cast(image, tf.float32)

        # Cast label data into int32
        label = tf.cast(features['label'], tf.int32)

        init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(500):
            example, l = sess.run([image, label])
            return_data = np.append(return_data, example)
            return_label = np.append(return_label, l)
            # print(return_data, return_label)
            # print (example, l)
        coord.request_stop()
        coord.join(threads)

        print(return_data)
        print(return_label)
    sess.close()
    print("done")
    return (return_data, return_label)

run()