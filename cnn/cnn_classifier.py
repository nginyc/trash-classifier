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
    'batch_size': 10,
    'learning_rate': 0.001,
    'train_steps': 100,
    'eval_steps': 1,
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

def get_feature_columns(params):
    feature_columns = {
        'images': tf.feature_column.numeric_column('images', (params['image_height'], params['image_width'], params['image_channels']))
    }
    return feature_columns

def model_fn(features, labels, mode, params):                                                                                         
    feature_columns = list(get_feature_columns(params).values())
    images = tf.feature_column.input_layer(
        features=features, feature_columns=feature_columns)
    images = tf.reshape(
        images, shape=(-1, params['image_height'], params['image_width'], params['image_channels']))
                                                                                                         
    logits = params['architecture'](images, params, mode)

    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        global_step = tf.train.get_or_create_global_step()
        label_indices = tf.argmax(input=logits, axis=1)
        # loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=label_indices, logits=logits)
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
        print("labels: ", labels.shape)
        print("Label indices: ", label_indices.shape)
        print("predicted: ", predicted_indices.shape)
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(label_indices, predicted_indices)
        }
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

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
    model_directory =  os.path.join(current_directory, "..", "model", "alexnet", params['model_name'])
    train_data_files = [os.path.join(current_directory, "..", "data", "tfrecords", "train.tfrecords")]
    test_data_files = [os.path.join(current_directory, "..", "data", "tfrecords", "test.tfrecords")]

    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=params['save_checkpoints_steps'],
        tf_random_seed=params['tf_random_seed'],
        model_dir=model_directory
    )
    
    # if not params['use_checkpoint']:
    #     print("Removing previous artifacts...")
    #     shutil.rmtree(model_directory, ignore_errors=True)

    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config, params=params)

    # tensors_to_log = {"probabilities": "softmax_tensor"}
    # logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    # train_input_fn = generate_input_fn(train_data_files, params, mode=tf.estimator.ModeKeys.TRAIN)
    # estimator.train(train_input_fn, max_steps=params['train_steps'], hooks=[logging_hook])
    
    test_input_fn = generate_input_fn(test_data_files, params, mode=tf.estimator.ModeKeys.EVAL)
    eval_results = estimator.evaluate(test_input_fn, steps=params['eval_steps'])
    print("Evaluation Results: {}".format(eval_results))

if __name__ == "__main__":
    tf.app.run()