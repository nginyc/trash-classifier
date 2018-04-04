from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from cnn_networks import *
from tfrecords_to_dataset import *

tf.logging.set_verbosity(tf.logging.DEBUG)

CNN_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(CNN_DIRECTORY, "..", "data", "tfrecords", "train.tfrecords")
TEST_PATH = os.path.join(CNN_DIRECTORY, "..", "data", "tfrecords", "test.tfrecords")
ALEXNET_MODEL_PATH = os.path.join(CNN_DIRECTORY, "..", "model", "alexnet")

def alexnet_model_fn(features, labels, mode, params):
    inputs = tf.reshape(features["image_data"], [-1, 256, 256, 3])
    logits = alexnet_layers_fn(inputs, mode)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=params.learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def run_alexnet_experiment(argv=None):
    """Run the training experiment."""
    # Define model parameters
    params = tf.contrib.training.HParams(
        n_classes=6,
        learning_rate=0.002,
        train_steps=5000,
        batch_size=32
    )

    # Set the run_config and the directory to save the model and stats
    run_config = tf.contrib.learn.RunConfig()
    run_config = run_config.replace(model_dir=ALEXNET_MODEL_PATH)

    tf.contrib.learn.learn_runner.run(
        experiment_fn=experiment_fn,  # First-class function
        run_config=run_config,  # RunConfig
        schedule="train_and_evaluate",  # What to run
        hparams=params  # HParams
    )

def experiment_fn(run_config, params):
    estimator = get_estimator(run_config, params)
    train_input_fn, train_input_hook = tfrecords_to_dataset(params.batch_size, TRAIN_PATH) 
    eval_input_fn, eval_input_hook = tfrecords_to_dataset(params.batch_size, TEST_PATH)
    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,  # Estimator
        train_input_fn=train_input_fn,  # First-class function
        eval_input_fn=eval_input_fn,  # First-class function
        train_steps=params.train_steps,  # Minibatch steps
        train_monitors=[train_input_hook],  # Hooks for training
        eval_hooks=[eval_input_hook],  # Hooks for evaluation
        eval_steps=None  # Use evaluation feeder until its empty
    )
    return experiment

def get_estimator(run_config, params):
    """Return the model as a Tensorflow Estimator object.
    Args:
         run_config (RunConfig): Configuration for Estimator run.
         params (HParams): hyperparameters.
    """
    return tf.estimator.Estimator(
        model_fn=alexnet_model_fn,  # First-class function
        params=params,  # HParams
        config=run_config  # RunConfig
    )

if __name__ == "__main__":  
    tf.app.run(main=run_alexnet_experiment)