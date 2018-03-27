import numpy as np
import sys
import cv2
import matplotlib
from sklearn import svm, model_selection, metrics
matplotlib.use('TkAgg')  # Fixs matplotlib issue in virtualenv
import matplotlib.pyplot as plt
import os
from imutils import paths

DIR_PATH = os.path.dirname(__file__)
DATASET_PATH = DIR_PATH + '/../data/garythung-trashnet'

CLASSES = [
    {
        'name': 'cardboard',
        'image_dir_path': DATASET_PATH + '/cardboard'
    },
    {
        'name': 'glass',
        'image_dir_path': DATASET_PATH + '/glass'
    }
]

def get_data():
    print('Extracting data...')
    X = []
    y = []

    # For every class, for every image in its images dataset, accumulate into list of features & labels
    for (i, clazz) in enumerate(CLASSES):
        image_paths = list(paths.list_images(clazz.get('image_dir_path')))
        for image_path in image_paths:
            image = cv2.imread(
                image_path,
                cv2.IMREAD_COLOR
            )
            x = image_to_feature_vector(image)
            X.append(x)
            y.append(i)

    return (X, y)

def image_to_feature_vector(image):
    # TODO Better feature extraction
    x = image.flatten()
    return x

def train(X, y):
    '''
    Returns (accuracy, confusion matrix)
    '''
    print('Training model...')
    X = np.array(X)
    y = np.array(y)
    
    # TODO N-fold cross validation
    (X_train, X_test, y_train, y_test) = model_selection.train_test_split(
	    X, y, test_size=0.25, random_state=666
    )
    
    model = svm.SVC(max_iter=1000, verbose=True)
    model.fit(X_train, y_train)
    y_test_predict = np.array(model.predict(X_test))

    accuracy = np.sum(y_test_predict == y_test) / y_test.size
    confusion_matrix = metrics.confusion_matrix(y_test_predict, y_test)

    return (accuracy, confusion_matrix)

def main():
    (X, y) = get_data()
    (accuracy, confusion_matrix) = train(X, y)

    print('\n')
    print('Accuracy: ' + str(accuracy))
    print('Confusion matrix:')
    print(confusion_matrix)

    return 0

if __name__ == '__main__':
    sys.exit(main())
