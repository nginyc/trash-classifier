from tfrecord_to_dataset import generate_input_fn
from cnn_architecture import *
import tensorflow as tf
import numpy as np
import sys
import os
import getopt
import shutil

tf.logging.set_verbosity(tf.logging.INFO)

alexnet_params = {
    'batch_size': 32,
    'learning_rate': 0.002,
    'train_steps': 300,
    'eval_steps': 100,
    'num_classes': 5,
    'image_height': 256,
    'image_width': 256,
    'image_channels': 3,
    'architecture': alexnet_architecture,
    'save_checkpoints_steps': 100,
    'use_checkpoint': False,
    'log_step_count_steps': 1,
    'logging_steps': 10,
    'tf_random_seed': 20170409,
    'model_name': 'alexnet_model'
}

zfnet_params = {
    'batch_size': 2,
    'learning_rate': 0.002,
    'train_steps': 10,
    'eval_steps': 11,
    'num_classes': 5,
    'image_height': 256,
    'image_width': 256,
    'image_channels': 3,
    'architecture': zfnet_layers_fn,
    'save_checkpoints_steps': 100,
    'use_checkpoint': False,
    'log_step_count_steps': 1,
    'logging_steps': 10,
    'tf_random_seed': 20170417,
    'model_name': 'zfnet_model'
}

architecture = {
    'alexnet': alexnet_params,
    'zfnet': zfnet_params
}

def model_fn(features, labels, mode, params):                                                                                                                                                                         
    logits = params['architecture'](features, params, mode)

    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        global_step = tf.train.get_or_create_global_step()
        label_indices = tf.argmax(input=labels, axis=1)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        tf.summary.scalar('cross_entropy', loss)

    predicted_indices = tf.argmax(input=logits, axis=1)
    probabilities = tf.nn.softmax(logits, name='softmax_tensor')
    predictions = {
        'classes': predicted_indices,
        'probabilities': probabilities
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(label_indices, predicted_indices),
            'confusion_matrix': eval_confusion_matrix(label_indices, predicted_indices, params)
        }
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

def eval_confusion_matrix(labels, predictions, params):
    with tf.variable_scope("eval_confusion_matrix"):
        con_matrix = tf.confusion_matrix(labels=labels, predictions=predictions, num_classes=params['num_classes'])
        con_matrix_sum = tf.Variable(
            tf.zeros(
                shape=(params['num_classes'], params['num_classes']), 
                dtype=tf.int32
            ),
            trainable=False,
            name="confusion_matrix_result",
            collections=[tf.GraphKeys.LOCAL_VARIABLES]
        )
        update_op = tf.assign_add(con_matrix_sum, con_matrix)
        return tf.convert_to_tensor(con_matrix_sum), update_op

def main(argv):
    try:
        opts, args = getopt.getopt(argv[1:], "ha:", ["help=", "architecture="])
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
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=params['logging_steps'])

    train_input_fn = generate_input_fn(train_data_files, params, mode=tf.estimator.ModeKeys.TRAIN)
    estimator.train(train_input_fn, max_steps=params['train_steps'], hooks=[logging_hook])
    
    test_input_fn = generate_input_fn(test_data_files, params, mode=tf.estimator.ModeKeys.EVAL)
    eval_results = estimator.evaluate(test_input_fn, steps=params['eval_steps'], hooks=[logging_hook])

    print("Accuracy: {}".format(eval_results['accuracy']))
    print("Confusion Matrix: \n{}".format(eval_results['confusion_matrix']))

if __name__ == "__main__":
    tf.app.run()