import numpy as np
from sklearn import svm, model_selection, metrics
import matplotlib
import os

from .visualize_features import visualize_features

'''
    SETTINGS (can be configured with environment variables)
'''
SVM_C_PARAM = float(os.environ.get('SVM_C_PARAM', 1))
MAX_ITERATIONS = int(os.environ.get('MAX_ITERATIONS', 10000))
KFOLD_SPLITS = int(os.environ.get('KFOLD_SPLITS', 5))
KFOLD_RANDOM_STATE = int(os.environ.get('KFOLD_RANDOM_STATE', 666))
IF_VISUALIZE_FEATURES = bool(os.environ.get('IF_VISUALIZE_FEATURES', False))

def train_and_test_svm(X, y):
    if IF_VISUALIZE_FEATURES:
        visualize_features(X, y)

    print('Training & testing model...')
    X = np.array(X)
    y = np.array(y)
    
    kf = model_selection.KFold(
        n_splits=KFOLD_SPLITS, random_state=KFOLD_RANDOM_STATE, shuffle=True
    )

    y_test_predict_comb = []
    y_test_comb = []
    y_train_predict_comb = []
    y_train_comb = []
    for (i, (train_indices, test_indices)) in enumerate(kf.split(X)):
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
        print('Training model for fold ' + str(i) + '...')
        model = svm.SVC(max_iter=MAX_ITERATIONS, C=SVM_C_PARAM)
        model.fit(X_train, y_train)
        print('Testing model for fold ' + str(i) + '...')
        y_train_predict = np.array(model.predict(X_train))
        y_test_predict = np.array(model.predict(X_test))
        (train_accuracy, _) = get_accuracy(y_train_predict, y_train)
        (test_accuracy, _) = get_accuracy(y_test_predict, y_test)
        print('Train accuracy: ' + str(train_accuracy))
        print('Test accuracy: ' + str(test_accuracy))
        y_train_predict_comb = np.concatenate((y_train_predict_comb, y_train_predict))
        y_train_comb = np.concatenate((y_train_comb, y_train))
        y_test_predict_comb = np.concatenate((y_test_predict_comb, y_test_predict))
        y_test_comb = np.concatenate((y_test_comb, y_test))

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
    accuracy = np.sum(y_predict == y) / y.size
    confusion_matrix = metrics.confusion_matrix(y_predict, y)
    return (accuracy, confusion_matrix)