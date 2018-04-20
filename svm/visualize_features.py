import numpy as np
from sklearn import svm, model_selection, metrics
from sklearn.decomposition import PCA 
import matplotlib
import os
matplotlib.use('TkAgg')  # Fixs matplotlib issue in virtualenv
import matplotlib.pyplot as plt

'''
    SETTINGS (can be configured with environment variables)
'''

def visualize_features(X, y):
    print('Running PCA on (X, y)...')
    reduced_X = PCA(n_components=2).fit_transform(X)
    print('Displaying dataset on scatterplot...')
    plt.figure(1)
    plt.clf()
    plt.scatter(reduced_X[:, 0], reduced_X[:, 1], c=y)
    plt.title('Dataset after feature extraction & PCA')
    plt.show()
