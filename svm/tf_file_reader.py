import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import  Image

# data_path = 'train.tfrecords'  # address to save the hdf5 file
# DATASET_PATH = os.environ.get('DATASET_PATH',
#     os.path.join(os.path.dirname(__file__) ,'../data/garythung-trashnet'))
DATASET_PATH =  os.path.join(os.getcwd(),'../data/garythung-trashnet')
print(DATASET_PATH)
# data_path = [DATASET_PATH + '/cardboard/train.tfrecords' , DATASET_PATH + '/trash/train.tfrecords']
data_path = [os.path.join(DATASET_PATH,'cardboard/train.tfrecords') , os.path.join(DATASET_PATH,'trash/train.tfrecords')]
# data_path = [os.path.join(DATASET_PATH,'trash/train.tfrecords')]
# with tf.Session() as sess:
def run():
    with tf.Session() as sess:
        print("start")
        feature = {'image': tf.FixedLenFeature([], tf.string),
                   'label': tf.FixedLenFeature([], tf.int64)}
        # Create a list of filenames and pass it to a queue
        print(data_path)
        filename_queue = tf.train.string_input_producer(data_path, num_epochs=1)
        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature)
        # Convert the image data from string back to the numbers
        image = tf.decode_raw(features['image'], tf.uint8)
        image = tf.cast(image, tf.float32)

        # image = tf.reshape(image, tf.stack([720,720,3]))
        # image.set_shape([720, 720, 3])

        # Cast label data into int32
        label = tf.cast(features['label'], tf.int32)
        # Reshape image data into the original shape
        init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        sess.run(init_op)
        print("Whats happ")
        # example, l = sess.run([image, label])
        # img = Image.fromarray(example, 'RGB')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(500):
            example, l = sess.run([image, label])
            # img = Image.fromarray(example, 'RGB')
            # img.save(DATASET_PATH + 'filename-train.jpeg')

            print (example, l)
        coord.request_stop()
        coord.join(threads)

        print("saved")
        print (example, l)

run()