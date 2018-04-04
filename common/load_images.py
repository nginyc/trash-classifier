import cv2
from imutils import paths
import os

'''
    SETTINGS (can be configured with environment variables)
'''
IMAGE_COUNT_PER_CLASS = int(os.environ.get('IMAGE_COUNT_PER_CLASS', 0))
DATASET_PATH = os.environ.get('DATASET_PATH', 
    os.path.dirname(__file__) + '/../data/garythung-trashnet')

CLASSES = [
    {
        'name': 'cardboard',
        'image_dir_path': DATASET_PATH + '/cardboard'
    },
    {
        'name': 'glass',
        'image_dir_path': DATASET_PATH + '/glass'
    },
    {
        'name': 'metal',
        'image_dir_path': DATASET_PATH + '/metal'
    },
    {
        'name': 'paper',
        'image_dir_path': DATASET_PATH + '/paper'
    },
    {
        'name': 'plastic',
        'image_dir_path': DATASET_PATH + '/plastic'
    },
    # {
    #     'name': 'trash',
    #     'image_dir_path': DATASET_PATH + '/trash'
    # }
]

'''
    Loads the image dataset from the filesystem in arrays suitable for training.
    Returns:
      (images, image_labels) where
        images: list of numpy ndarrays of shape (image_height, image_width, num_channels=3) 
        image_labels: list of labels as class indices corresponding to each row of `images`
'''
def load_images():
    print('Loading image data...')
    images = []
    image_labels = []

    # For every class, for every image in its images dataset, accumulate into list of features & labels
    for (i, clazz) in enumerate(CLASSES):
        image_paths = list(paths.list_images(clazz.get('image_dir_path')))
        image_paths = image_paths[0:IMAGE_COUNT_PER_CLASS] if IMAGE_COUNT_PER_CLASS else image_paths
        for image_path in image_paths:
            image = cv2.imread(
                image_path,
                cv2.IMREAD_COLOR
            )
            images.append(image)
            image_labels.append(i)
    

    return (images, image_labels)