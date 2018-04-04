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

ALEXNET_MODEL_PATH = os.path.join(CNN_DIRECTORY, "..", "model", "alexnet")

class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)

def tfrecords_to_dataset(batch_size, filename):
    dataset = tf.data.TFRecordDataset([filename])
    iterator_initializer_hook = IteratorInitializerHook()

    def parser(record):
        keys_to_features = {
            "image_data": tf.FixedLenFeature((), tf.string, default_value=""),
            "label": tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64))
        }
        parsed = tf.parse_single_example(record, keys_to_features)

        # Perform additional preprocessing on the parsed data.
        image = tf.image.decode_jpeg(parsed["image_data"])
        image = tf.image.resize_image_with_crop_or_pad(image, 256, 256)
        label = tf.cast(parsed["label"], tf.int32)

        return {"image_data": image}, label

    def train_inputs():
        with tf.name_scope('Training_data'):
            dataset = dataset.map(parser)
            dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.batch(batch_size)
            dataset = dataset.repeat(None)  # Infinite iterations
            iterator = dataset.make_one_shot_iterator()
            features, labels = iterator.get_next()

            iterator_initializer_hook.iterator_initializer_func = lambda sess: sess.run(
                iterator.initializer,
                feed_dict={images_placeholder: images, labels_placeholder: labels})
            return features, labels

    return train_inputs, iterator_initializer_hook

def alexnet_model_fn(features, lablels, mode):
    inputs = tf.reshape(features["image_data"], [-1, 256, 256, 3])
    logits = alexnet_layers_fn(inputs)

def run_experiment(argv=None):
    """Run the training experiment."""
    # Define model parameters
    params = tf.contrib.training.HParams(
        learning_rate=0.002,
        n_classes=6,
        train_steps=5000
    )

    # Set the run_config and the directory to save the model and stats
    run_config = tf.contrib.learn.RunConfig()
    run_config = run_config.replace(model_dir=FLAGS.model_dir)

    tf.contrib.learn.learn_runner.run(
        experiment_fn=experiment_fn,  # First-class function
        run_config=run_config,  # RunConfig
        schedule="train_and_evaluate",  # What to run
        hparams=params  # HParams
    )

        

if __name__ == "__main__":  
    tf.app.run(main=run_experiment)