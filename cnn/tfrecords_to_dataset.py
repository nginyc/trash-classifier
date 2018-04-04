from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)

def tfrecords_to_dataset(batch_size, filename):
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
        image = tf.cast(image, tf.float32)
        label = tf.cast(parsed["label"], tf.int32)

        return {"image_data": image}, label

    def get_inputs():
        with tf.name_scope('Training_data'):
            dataset = tf.data.TFRecordDataset([filename])
            dataset = dataset.map(parser)
            dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.batch(batch_size)
            dataset = dataset.repeat(None)  # Infinite iterations
            # iterator = dataset.make_one_shot_iterator()
            iterator = dataset.make_initializable_iterator()
            features, labels = iterator.get_next()
            filenames = tf.placeholder(tf.string, shape=[None])
            iterator_initializer_hook.iterator_initializer_func = lambda sess: sess.run(
                iterator.initializer,
                feed_dict={filenames: [filename]})
            return features, labels

    return get_inputs, iterator_initializer_hook