# Multi-class (Nonlinear) SVM Example
# ----------------------------------
#
# This function wll illustrate how to
# implement the gaussian kernel with
# multiple classes on the iris dataset.
#
# Gaussian Kernel:
# K(x1, x2) = exp(-gamma * abs(x1 - x2)^2)
#
# X : (Sepal Length, Petal Width)
# Y: (I. setosa, I. virginica, I. versicolor) (3 classes)
#
# Basic idea: introduce an extra dimension to do
# one vs all classification.
#
# The prediction of a point will be the category with
# the largest margin or distance to boundary.

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
from sklearn import datasets
from tensorflow.python.framework import ops
from common import tfrecords_to_dataset
import tf_file_reader
import scipy.cluster.vq as vq

tf.logging.set_verbosity(tf.logging.DEBUG)

SVM_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

ops.reset_default_graph()

# Create graph
# sess = tf.Session()
# reads the data_set as of now just to check if this is running
# only reads around 1000 images
# NOTE :  PLEASE USE MY VERSION OF TF generator and extractor as of now will change it correctly later SORRY!!
dataset = tf_file_reader.run()


# if you want to try the SIFT feature, may be slower
# sift = cv2.xfeatures2d_SIFT.create()
# kp , des = sift.detectAndCompute(img,None)


with tf.Session() as sess:
    print("start")
    init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    feature = list()
    onlydes = list()
    for data in (dataset):
        # need to transform data from a 1D array to an acceptable form by CV2
        img = data[0][0:512 * 384].reshape(384, 512)
        # orb to get the image keypoints
        orb = cv2.ORB.create()
        # orb to get description vector size 32 of the keypoints
        kp = orb.detect(img,None)
        kp, des = orb.compute(img, kp)
        feature.append((des, data[1]))
        onlydes.append(des)
    print(feature[0][0].shape)
    coord.request_stop()
    coord.join(threads)
    sess.close()
nkeys = len(onlydes)
print(nkeys)

array = np.zeros((nkeys * 1000, 32))
pivot = 0
ignore= 0
# onlydes = np.array(onlydes)
for key in range(len(onlydes)):
    print(key)
    value = onlydes[key]
    if value is None:
        ignore += 1
        continue
    nelements = value.shape[0]
    while pivot + nelements > array.shape[0]:
        padding = np.zeros_like(array)
        array = np.vstack((array, padding))
    array[pivot:pivot + nelements] = value
    pivot += nelements
    print("success")
print('Ignored Imges ' ,ignore) # imaeges with no substantial description -- strange, need to look at this
array = np.resize(array, (pivot, 32))

nfeatures = array.shape[0]
nclusters = int(np.sqrt(nfeatures))
print nclusters
codebook, distortion = vq.kmeans(array,
                                 nclusters,
                                 thresh=1)
print "CodeBook"
print codebook

print codebook.shape
print(codebook[0])


# compute a histogram of bag of words
def computeHistograms(codebook, descriptors):
    code, dist = vq.vq(descriptors, codebook)
    histogram_of_words, bin_edges = np.histogram(code,
                                          bins=range(
                                              codebook.shape[0] + 1),
                                          normed=True)
    return  histogram_of_words

hist = np.array([computeHistograms(codebook, f) for f in onlydes if f is not None ]) # computeHistograms(codebook, onlydes)
print("Hist")

print(len(hist))
print(hist[0])
print(len(hist[0]))
print(hist.shape)


# onlydes
# TODO : Follwo the example commented out in this part
# Implement SVM using the code below to best fit out need

# TODO : Add feature selection,
# basically from the features extracted through openCV fine tune them

# TODO : We are going to start with Gussaian RBF kernel
# can look at different kernels as well






# Load the data
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
# iris = datasets.load_iris()


# these are the featues
# x_vals = np.array([[x[0], x[3]] for x in iris.data])
# classes correspond to which being true
# y_vals1 = np.array([1 if y == 0 else -1 for y in iris.target])
# y_vals2 = np.array([1 if y == 1 else -1 for y in iris.target])
# y_vals3 = np.array([1 if y == 2 else -1 for y in iris.target])

# forms a 2d array
#[[0, 0, 1],
#[0, 1, 0],
#[1, 0, 0]]

# feature vertex for each class
# each class has class1_X = feature 1 value
# y_vals = np.array([y_vals1, y_vals2, y_vals3])
# class1_x = [x[0] for i, x in enumerate(x_vals) if iris.target[i] == 0]
# class1_y = [x[1] for i, x in enumerate(x_vals) if iris.target[i] == 0]
# class2_x = [x[0] for i, x in enumerate(x_vals) if iris.target[i] == 1]
# class2_y = [x[1] for i, x in enumerate(x_vals) if iris.target[i] == 1]
# class3_x = [x[0] for i, x in enumerate(x_vals) if iris.target[i] == 2]
# class3_y = [x[1] for i, x in enumerate(x_vals) if iris.target[i] == 2]
#
# Declare batch size
# batch_size = 50
# #
# # Initialize placeholders
# x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
# y_target = tf.placeholder(shape=[3, None], dtype=tf.float32)
# prediction_grid = tf.placeholder(shape=[None, 2], dtype=tf.float32)
# #
# # # Create variables for svm
# b = tf.Variable(tf.random_normal(shape=[3, batch_size]))
#
#  # Gaussian (RBF) kernel
# gamma = tf.constant(-10.0)
# dist = tf.reduce_sum(tf.square(x_data), 1)
# dist = tf.reshape(dist, [-1, 1]) # basically say that transform the tensor to [[one value], [1v] ,[1v]]
# sq_dists = tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data))) # x,2 2,x == x,x
# my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))
#
#  # Declare function to do reshape/batch multiplication
# def reshape_matmul(mat):
#     v1 = tf.expand_dims(mat, 1)
#     v2 = tf.reshape(v1, [3, batch_size, 1])
#     return (tf.matmul(v2, v1))
#
#
# # Compute SVM Model
# first_term = tf.reduce_sum(b)
# b_vec_cross = tf.matmul(tf.transpose(b), b)
# y_target_cross = reshape_matmul(y_target)
#
# second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)), [1, 2])
# loss = tf.reduce_sum(tf.negative(tf.subtract(first_term, second_term)))
#
# # Gaussian (RBF) prediction kernel
# rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
# rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1, 1])
# pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))),
#                       tf.transpose(rB))
# pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))
#
# prediction_output = tf.matmul(tf.multiply(y_target, b), pred_kernel)
# prediction = tf.arg_max(prediction_output - tf.expand_dims(tf.reduce_mean(prediction_output, 1), 1), 0)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y_target, 0)), tf.float32))

# Declare optimizer
# my_opt = tf.train.GradientDescentOptimizer(0.01)
# train_step = my_opt.minimize(loss)

# Initialize variables
# init = tf.global_variables_initializer()
# sess.run(init)

# Training loop
# loss_vec = []
# batch_accuracy = []
# for i in range(100):
#     rand_index = np.random.choice(len(x_vals), size=batch_size) # randomly choose batch size of the examples
#     rand_x = x_vals[rand_index] # random rows of weights
#     rand_y = y_vals[:, rand_index] # random colmns getting assignment
#     sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
#
#     temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
#     loss_vec.append(temp_loss)
#
#     acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x,
#                                              y_target: rand_y,
#                                              prediction_grid: rand_x})
#     batch_accuracy.append(acc_temp)
#
#     if (i + 1) % 25 == 0:
#         print('Step #' + str(i + 1))
#         print('Loss = ' + str(temp_loss))
#
# # Create a mesh to plot points in
# x_min, x_max = x_vals[:, 0].min() - 1, x_vals[:, 0].max() + 1
# y_min, y_max = x_vals[:, 1].min() - 1, x_vals[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
#                      np.arange(y_min, y_max, 0.02))
# grid_points = np.c_[xx.ravel(), yy.ravel()]
# grid_predictions = sess.run(prediction, feed_dict={x_data: rand_x,
#                                                    y_target: rand_y,
#                                                    prediction_grid: grid_points})
# grid_predictions = grid_predictions.reshape(xx.shape)
#


# # Plot points and grid


# plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.Paired, alpha=0.8)
# plt.plot(class1_x, class1_y, 'ro', label='I. setosa')
# plt.plot(class2_x, class2_y, 'kx', label='I. versicolor')
# plt.plot(class3_x, class3_y, 'gv', label='I. virginica')
# plt.title('Gaussian SVM Results on Iris Data')
# plt.xlabel('Pedal Length')
# plt.ylabel('Sepal Width')
# plt.legend(loc='lower right')
# plt.ylim([-0.5, 3.0])
# plt.xlim([3.5, 8.5])
# plt.show()
#
# # Plot batch accuracy
# plt.plot(batch_accuracy, 'k-', label='Accuracy')
# plt.title('Batch Accuracy')
# plt.xlabel('Generation')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.show()
#
# # Plot loss over time
# plt.plot(loss_vec, 'k-')
# plt.title('Loss per Generation')
# plt.xlabel('Generation')
# plt.ylabel('Loss')
# plt.show()