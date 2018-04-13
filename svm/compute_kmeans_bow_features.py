import os
import numpy as np
from sklearn import svm, model_selection, metrics, cluster

KMEANS_CLUSTERS = int(os.environ.get('KMEANS_CLUSTERS', 128))
KMEANS_JOBS = int(os.environ.get('KMEANS_JOBS', -2))
IF_SQRT_KEYPOINTS_KMEANS_CLUSTERS = bool(os.environ.get('IF_SQRT_KEYPOINTS_KMEANS_CLUSTERS', False))
IF_INSTANCES_COUNT_KEYPOINTS_KMEANS_CLUSTERS = bool(os.environ.get('IF_INSTANCES_COUNT_KEYPOINTS_KMEANS_CLUSTERS', False))
IF_BINARY_KMEANS_BOW_FEATURES = bool(os.environ.get('IF_BINARY_KMEANS_BOW_FEATURES', False))
IF_NORMALIZED_KMEANS_BOW_FEATURES = bool(os.environ.get('IF_NORMALIZED_KMEANS_BOW_FEATURES', False))

# Ref: https://dsp.stackexchange.com/questions/5979/image-classification-using-sift-features-and-svm
def compute_kmeans_bow_features(images_keypoints_train, images_keypoints_test):
    # Train KMeans on train set
    flattened_image_keypoints = [point for image_keypoints in images_keypoints_train for point in image_keypoints]
    num_clusters = KMEANS_CLUSTERS
    if IF_INSTANCES_COUNT_KEYPOINTS_KMEANS_CLUSTERS:
      num_clusters = len(images_keypoints_train)
    elif IF_SQRT_KEYPOINTS_KMEANS_CLUSTERS:
      num_clusters = int(np.sqrt(len(flattened_image_keypoints)))
    print('Computing KMeans clusters with n_clusters=' + str(num_clusters) + '...')
    # n_jobs=-2 makes it runn all all cores except 1. As compated to when it was 1 and ran sequentially :(
    kmeans = cluster.KMeans(n_clusters=num_clusters, n_jobs=KMEANS_JOBS)
    kmeans.fit(flattened_image_keypoints)

    # Compute BoW features for train & test set
    X_train = predict_bow_features(kmeans, images_keypoints_train)
    X_test = predict_bow_features(kmeans, images_keypoints_test)
    
    if IF_BINARY_KMEANS_BOW_FEATURES:
        print('Converting to binary BoW features...')
        X_train = (X_train > 0).astype(int)
        X_test = (X_test > 0).astype(int)

    if IF_NORMALIZED_KMEANS_BOW_FEATURES:
        print('Normalizing BoW features...')
        # Normalizing with means rather than sums that reduces precisions
        cluster_means = X_train.mean(axis=0)
        X_train = X_train / cluster_means
        X_test = X_test / cluster_means
        
    return (X_train, X_test)

def predict_bow_features(kmeans, image_keypoints):
    num_clusters = kmeans.n_clusters
    X = np.empty((len(image_keypoints), num_clusters))
    for (i, image_keypoints) in enumerate(image_keypoints):
        clusters = np.array(kmeans.predict(image_keypoints) if len(image_keypoints) > 0 else [])
        # Get cluster number histogram as feature vector
        cluster_vector = [(clusters == i).sum() for i in range(0, num_clusters)]
        X[i] = cluster_vector
    return X