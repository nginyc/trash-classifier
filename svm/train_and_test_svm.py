import numpy as np
from sklearn import svm, model_selection, metrics
import matplotlib
import os
from sklearn.cluster import KMeans

'''
    SETTINGS (can be configured with environment variables)
'''
IF_BINARY_FEATURES = bool(os.environ.get('IF_BINARY_FEATURES', False))
SVM_C_PARAM = float(os.environ.get('SVM_C_PARAM', 1))
MAX_ITERATIONS = int(os.environ.get('MAX_ITERATIONS', 10000))
KFOLD_SPLITS = int(os.environ.get('KFOLD_SPLITS', 5))
KFOLD_RANDOM_STATE = int(os.environ.get('KFOLD_RANDOM_STATE', 666))
IF_VISUALIZE_FEATURES = bool(os.environ.get('IF_VISUALIZE_FEATURES', False))
IF_SQRT_KEYPOINTS_KMEANS_CLUSTERS = bool(os.environ.get('IF_SQRT_KEYPOINTS_KMEANS_CLUSTERS', False))

def train_and_test_svm(X, y):
    if IF_VISUALIZE_FEATURES:
        visualize_features(X, y)

    print('Training & testing model...')
    X = np.array(X)
    y = np.array(y)

    kf = model_selection.KFold(
        n_splits=KFOLD_SPLITS, random_state=KFOLD_RANDOM_STATE, shuffle=True
    )
    # split_index = int(len(X) * 0.70)
    train_index = np.random.choice(len(X), size=int(len(X) * 0.7), replace=False)

    y_test_predict_comb = []
    y_test_comb = []
    y_train_predict_comb = []
    y_train_comb = []
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = np.delete(X, train_index)
    y_test = np.delete(y, train_index)
    cluster_vectors_train, kmeans = compute_kmeans_cluster_vectors(X_train)
    cluster_vectors_test = []
    for image_keypoints in X_test:
        clusters = np.array(kmeans.predict(image_keypoints) if len(image_keypoints) > 0 else [])

        # Get cluster number histogram as feature vector
        #num_clusters = len(X_train)
        num_clusters = kmeans.cluster_centers_.shape[0]
        cluster_vector = [(clusters == i).sum() for i in range(0, num_clusters)]
        if IF_BINARY_FEATURES:
            cluster_vector = [1 if x > 0 else 0 for x in cluster_vector]

        cluster_vectors_test.append(cluster_vector)

    print('Training model for fold 1' + '...')
    model = svm.SVC(max_iter=MAX_ITERATIONS, C=SVM_C_PARAM, gamma=0.5)
    model.fit(cluster_vectors_train, y_train)
    print('Testing model for fold 1' + '...')
    y_train_predict = np.array(model.predict(cluster_vectors_train))
    y_test_predict = np.array(model.predict(cluster_vectors_test))
    (train_accuracy, _) = get_accuracy(y_train_predict, y_train)
    (test_accuracy, _) = get_accuracy(y_test_predict, y_test)
    print('Train accuracy: ' + str(train_accuracy))
    print('Test accuracy: ' + str(test_accuracy))
    y_train_predict_comb = np.concatenate((y_train_predict_comb, y_train_predict))
    y_train_comb = np.concatenate((y_train_comb, y_train))
    y_test_predict_comb = np.concatenate((y_test_predict_comb, y_test_predict))
    y_test_comb = np.concatenate((y_test_comb, y_test))
    # for (i, (train_indices, test_indices)) in enumerate(kf.split(X)):
    #     X_train = X[train_indices]
    #     y_train = y[train_indices]
    #     X_test = X[test_indices]
    #     y_test = y[test_indices]
    #     print('Training model for fold ' + str(i) + '...')
    #     model = svm.SVC(max_iter=MAX_ITERATIONS, C=SVM_C_PARAM, gamma=0.5)
    #     model.fit(X_train, y_train)
    #     print('Testing model for fold ' + str(i) + '...')
    #     y_train_predict = np.array(model.predict(X_train))
    #     y_test_predict = np.array(model.predict(X_test))
    #     (train_accuracy, _) = get_accuracy(y_train_predict, y_train)
    #     (test_accuracy, _) = get_accuracy(y_test_predict, y_test)
    #     print('Train accuracy: ' + str(train_accuracy))
    #     print('Test accuracy: ' + str(test_accuracy))
    #     y_train_predict_comb = np.concatenate((y_train_predict_comb, y_train_predict))
    #     y_train_comb = np.concatenate((y_train_comb, y_train))
    #     y_test_predict_comb = np.concatenate((y_test_predict_comb, y_test_predict))
    #     y_test_comb = np.concatenate((y_test_comb, y_test))

    (train_accuracy_comb, train_confusion_matrix_comb) = get_accuracy(y_train_predict_comb, y_train_comb)
    (test_accuracy_comb, test_confusion_matrix_comb) = get_accuracy(y_test_predict_comb, y_test_comb)
    print('\n')
    print('Combined train accuracy: ' + str(train_accuracy_comb))
    print('Combined train confusion matrix:')
    print(train_confusion_matrix_comb)
    print('Combined test accuracy: ' + str(test_accuracy_comb))
    print('Combined test confusion matrix:')
    print(test_confusion_matrix_comb)

def get_accuracy(y_predict, y):
    accuracy = np.sum(y_predict == y).__float__() / float(y.size)
    confusion_matrix = metrics.confusion_matrix(y_predict, y)
    return (accuracy, confusion_matrix)


def compute_kmeans_cluster_vectors(image_keypoint_lists):
    flattened_image_keypoints = [point for image_keypoints in image_keypoint_lists for point in image_keypoints]

    # num_clusters = KMEANS_CLUSTERS
    num_clusters = len(image_keypoint_lists)
    if IF_SQRT_KEYPOINTS_KMEANS_CLUSTERS:
      num_clusters = int(np.sqrt(len(flattened_image_keypoints)))
    print('Computing KMeans clusters with n_clusters=' + str(num_clusters) + '...')
    # n_jobs=-2 makes it runn all all cores except 1. As compated to when it was 1 and ran sequentially :(
    kmeans = KMeans(n_clusters=num_clusters, n_jobs=-2)
    kmeans.fit(flattened_image_keypoints)

    cluster_vectors = []
    for image_keypoints in image_keypoint_lists:
        clusters = np.array(kmeans.predict(image_keypoints) if len(image_keypoints) > 0 else [])

        # Get cluster number histogram as feature vector
        cluster_vector = [(clusters == i).sum() for i in range(0, num_clusters)]
        if IF_BINARY_FEATURES:
            # print("Using Binary Features")
            cluster_vector = [1 if x > 0 else 0 for x in cluster_vector]

        cluster_vectors.append(cluster_vector)

    return cluster_vectors, kmeans
