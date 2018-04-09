from tfrecord_to_dataset import generate_input_fn
from cnn_architecture import *
import tensorflow as tf
import numpy as np
import sys
import os
import getopt

alexnet_params = {
    'batch_size': 10,
    'learning_rate': 0.001,
    'train_steps': 1000,
    'eval_steps': None,
    'num_classes': 6,
    'image_height': 256,
    'image_width': 256,
    'image_channels': 3,
    'architecture': alexnet_architecture,
    'save_checkpoints_steps': 100,
    'use_checkpoint': False,
    'tf_random_seed': 20170409,
    'model_name': 'alexnet_model'
}

architecture = {
    'alexnet': alexnet_params
}

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "ha:", ["help", "architecture="])
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                print("{} -a <architecture>".format(argv[0]))
                sys.exit()
            elif opt in ("-a", "--architecture"):
                params = architecture[arg]
    except getopt.GetoptError:
        print("{} -a <architecture>".format(argv[0]))
        sys.exit()

    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_directory =  os.path.join(current_directory, "..", "model", "alexnet", params.model_name)
    train_data_files = [os.path.join(current_directory, "..", "data", "tfrecords", "train.tfrecords")]
    test_data_files = [os.path.join(current_directory, "..", "data", "tfrecords", "test.tfrecords")]

    # run_config = tf.estimator.RunConfig(
    #     save_checkpoints_steps=FLAGS.save_checkpoints_steps,
    #     tf_random_seed=FLAGS.tf_random_seed,
    #     model_dir=model_dir
    # )

if __name__ == "__main__":
    tf.app.run()