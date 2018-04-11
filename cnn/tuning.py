import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math

# from tf.keras.models import Sequential  # This does not work!
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D, Dropout
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.layers import BatchNormalization

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.plots import plot_histogram, plot_objective_2D
from skopt.utils import use_named_args

from cnn_classifier import *
import os
from tfrecord_to_dataset import generate_input_fn
from cnn_architecture import *
import shutil

path_best_model = '19_best_model.keras'
best_accuracy = 0.0

# search dimension for learning rate
# return int k for 1ek
dim_learning_rate = Real(low=1e-6, high=1e-2, prior='log-uniform',
                         name='learning_rate')

# search dimension for number of dense layers
# return int n layers
# dim_num_dense_layers = Integer(low=1, high=5, name='num_dense_layers')

# search dimension for number of nodes for each dense layer
# return int n nodes
# dim_num_dense_nodes = Integer(low=5, high=512, name='num_dense_nodes')

# search dimension for activation function
# return relu or sigmoid
# dim_activation = Categorical(categories=['relu', 'sigmoid'],
#                              name='activation')

dimensions = [dim_learning_rate]

default_parameters = [0.002]

# logger for training progress
def log_dir_name(learning_rate):
# , num_dense_layers,
#                  num_dense_nodes, activation):

    # The dir-name for the TensorBoard log-dir.
    # s = "./19_logs/lr_{0:.0e}_layers_{1}_nodes_{2}_{3}/"
    s = "./19_logs/lr_{0:.0e}/"

    # Insert all the hyper-parameters in the dir-name.
    log_dir = s.format(learning_rate)
                       # num_dense_layers,
                       # num_dense_nodes,
                       # activation)

    return log_dir

def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

def plot_example_errors(cls_pred):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Boolean array whether the predicted class is incorrect.
    incorrect = (cls_pred != data.test.cls)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.test.images[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.test.cls[incorrect]
    
    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])

@use_named_args(dimensions=dimensions)
def _fitness(learning_rate):
    params = alexnet_params
    params['learning_rate'] = learning_rate

    # from main in cnn_classifier
    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_directory =  os.path.join(current_directory, "..", "model", params['model_name'])
    train_data_files = [os.path.join(current_directory, "..", "data", "tfrecords", "train.tfrecords")]
    test_data_files = [os.path.join(current_directory, "..", "data", "tfrecords", "test.tfrecords")]

    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=params['save_checkpoints_steps'],
        tf_random_seed=params['tf_random_seed'],
        model_dir=model_directory,
        log_step_count_steps=params['log_step_count_steps']
    )
    
    if not params['use_checkpoint']:
        print("Removing previous artifacts...")
        shutil.rmtree(model_directory, ignore_errors=True)

    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config, params=params)

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = generate_input_fn(train_data_files, params, mode=tf.estimator.ModeKeys.TRAIN)
    estimator.train(train_input_fn, max_steps=params['train_steps'], hooks=[logging_hook])
    
    test_input_fn = generate_input_fn(test_data_files, params, mode=tf.estimator.ModeKeys.EVAL)
    eval_results = estimator.evaluate(test_input_fn, steps=params['eval_steps'], hooks=[logging_hook])

    # skopt finds minimum, so we invert the accuracy
    return -eval_results['accuracy']


# test run
# _fitness(x=default_parameters)

# run model
search_result = gp_minimize(func=_fitness,
                            dimensions=dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls=40,
                            x0=default_parameters)

print(search_result.x)
space = search_result.space
print(space.point_to_dict(search_result.x))
print(search_result.fun)
plot_convergence(search_result)