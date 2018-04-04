import tensorflow as tf
import numpy as np
import os
import skimage.io as io

CNN_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIRECTORY = os.path.join(CNN_DIRECTORY, "..", "data", "garythung-trashnet")
TFRECORD_DIRECTORY = os.path.join(CNN_DIRECTORY, "..", "data", "tfrecords")

TRAIN_FILENAME = "train.tfrecords"
TEST_FILENAME = "test.tfrecords"

CATEGORIES = {
    'cardboard': 1,
    'glass': 2,
    'metal': 3,
    'paper': 4,
    'plastic': 5,
    'trash': 6 
}

def get_images_and_labels(file_dir, categories):
    images = []
    subfolders = []
    for root, sub_folders, files in os.walk(file_dir):
        for name in files:
            images.append(os.path.join(root, name))
        for name in sub_folders:
            subfolders.append(os.path.join(root, name))

    labels = []
    for subfolder in subfolders:        
        no_images = len(os.listdir(subfolder))
        category = subfolder.split('/')[-1]
        labels = np.append(labels, [categories[category]] * no_images)

    temp = np.array([images, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]

    return image_list, label_list

def int64_feature(value):
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def split(images, labels, ratio):
    cutoff = int(ratio * len(labels))

    train_images = images[:cutoff]
    train_labels = labels[:cutoff]

    test_images = images[cutoff:]
    test_labels = labels[cutoff:]

    return train_images, train_labels, test_images, test_labels


def convert_to_tfrecord(images, labels, save_dir, name):
    filename = os.path.join(save_dir, name + '.tfrecords')

    writer = tf.python_io.TFRecordWriter(filename)
    print('\nTransform start......')
    for i in np.arange(0, len(labels)):
        try:
            image = io.imread(images[i]) # type(image) must be array!
            image_raw = image.tostring()
            label = int(labels[i])
            example = tf.train.Example(features=tf.train.Features(feature={
                'label':int64_feature(label),
                'image_raw': bytes_feature(image_raw)
            }))
            writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:', images[i])
            print('error: %s' %e)
            print('Skip it!\n')
    writer.close()
    print('Transform done!')

def main(argv):
    images, labels = get_images_and_labels(IMAGE_DIRECTORY, CATEGORIES)
    train_images, train_labels, test_images, test_labels = split(images, labels, 0.7)
    convert_to_tfrecord(train_images, train_labels, TFRECORD_DIRECTORY, TRAIN_FILENAME)
    convert_to_tfrecord(test_images, test_labels, TFRECORD_DIRECTORY, TEST_FILENAME)

if __name__ == "__main__":
    tf.app.run()