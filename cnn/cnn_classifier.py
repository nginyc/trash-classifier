from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from cnn_networks import *

CNN_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(CNN_DIRECTORY, "..", "data", "tfrecords", "train.tfrecords")
TEST_PATH = os.path.join(CNN_DIRECTORY, "..", "data", "tfrecords", "test.tfrecords")

def dataset_input_fn(filename):
  dataset = tf.data.TFRecordDataset([filename])

  def parser(record):
    keys_to_features = {
        "image_data": tf.FixedLenFeature((), tf.string, default_value=""),
        "label": tf.FixedLenFeature((), tf.int64,
                                    default_value=tf.zeros([], dtype=tf.int64)),
    }
    parsed = tf.parse_single_example(record, keys_to_features)

    # Perform additional preprocessing on the parsed data.
    image = tf.image.decode_jpeg(parsed["image_data"])
    image = tf.image.resize_image_with_crop_or_pad(image, 256, 256)
    label = tf.cast(parsed["label"], tf.int32)

    return { "image_data": image }, label

  dataset = dataset.map(parser)
  dataset = dataset.shuffle(buffer_size=10000)
  dataset = dataset.batch(32)
  dataset = dataset.repeat(num_epochs)
  iterator = dataset.make_one_shot_iterator()

  features, labels = iterator.get_next()
  return features, labels

def main(argv):
    print(TRAIN_PATH)
    print("Running cnn.")

if __name__ == "__main__":  
    tf.app.run()