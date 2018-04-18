import os
import numpy as np
from sklearn import svm, model_selection, metrics, cluster, preprocessing

KMEANS_CLUSTERS = os.environ.get('KMEANS_CLUSTERS', None) # int | "sqrt_keypoints" | "instances_count" | "average_keypoints"
KMEANS_BOW_FEATURES_NORMALIZATION = os.environ.get('KMEANS_BOW_FEATURES_NORMALIZATION', None) # "binary" | "l1" | "minmax"
KMEANS_JOBS = int(os.environ.get('KMEANS_JOBS', 4))
KMEANS_MINI_BATCH_SIZE = os.environ.get('KMEANS_MINI_BATCH_SIZE', None)

# Ref: https://dsp.stackexchange.com/questions/5979/image-classification-using-sift-features-and-svm
def compute_kmeans_bow_features(images_keypoints_train, images_keypoints_test):
    # Train KMeans on train set
    flattened_image_keypoints = [point for image_keypoints in images_keypoints_train for point in image_keypoints]

    if KMEANS_CLUSTERS is None:
        num_clusters = 128
    elif KMEANS_CLUSTERS == 'sqrt_keypoints':
        num_clusters = int(np.sqrt(len(flattened_image_keypoints)))
    elif KMEANS_CLUSTERS == 'instances_count':
        num_clusters = len(images_keypoints_train)
    elif KMEANS_CLUSTERS == 'average_keypoints':
        num_clusters = (len(flattened_image_keypoints) / len(images_keypoints_train))
    else:
        num_clusters = int(KMEANS_CLUSTERS)
    # n_jobs=-2 makes it runn all all cores except 1. As compated to when it was 1 and ran sequentially :(

    if KMEANS_MINI_BATCH_SIZE is None:
        print('Running KMeans with n_clusters=' + str(num_clusters) + '...')
        kmeans = cluster.KMeans(n_clusters=num_clusters, n_jobs=KMEANS_JOBS)
    else:
        batch_size = int(KMEANS_MINI_BATCH_SIZE)
        print('Running Mini-batch KMeans with n_clusters=' + str(num_clusters) + ', batch_size=' + str(batch_size) + '...')
        kmeans = cluster.MiniBatchKMeans(n_clusters=num_clusters, batch_size=batch_size)

    kmeans.fit(flattened_image_keypoints)

    # Compute BoW features for train & test set
    X_train = predict_bow_features(kmeans, images_keypoints_train)
    X_test = predict_bow_features(kmeans, images_keypoints_test)

    if KMEANS_BOW_FEATURES_NORMALIZATION is None:
        pass
    elif KMEANS_BOW_FEATURES_NORMALIZATION == 'binary':
        print('Converting to binary BoW features...')
        X_train = (X_train > 0).astype(int)
        X_test = (X_test > 0).astype(int)
    elif KMEANS_BOW_FEATURES_NORMALIZATION == 'l1':
        print('L1 normalizing BoW features...')
        norms = np.linalg.norm(X_train, axis=0)
        X_train = X_train / norms
        X_test = X_test / norms
    elif KMEANS_BOW_FEATURES_NORMALIZATION == 'min_max':
        print('Min-max normalizing BoW features...')
        max = X_train.max(axis=0)
        min = X_train.min(axis=0)
        X_train = (X_train - min) / (max - min)
        X_test = (X_test - min) / (max - min)
    else:
        raise 'Invalid KMEANS_BOW_FEATURES_NORMALIZATION value'
        
    return (X_train, X_test)

def predict_bow_features(kmeans, image_keypoints):
    num_clusters = kmeans.n_clusters
    X = []
    for (i, image_keypoints) in enumerate(image_keypoints):
        clusters = np.array(kmeans.predict(image_keypoints) if len(image_keypoints) > 0 else [])
        # Get cluster number histogram as feature vector
        cluster_vector = [(clusters == i).sum() for i in range(0, num_clusters)]
        X.append(cluster_vector)
    return np.array(X)