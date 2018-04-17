import tensorflow as tf
import numpy as np
import os

CNN_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIRECTORY = os.path.join(CNN_DIRECTORY, "..", "data", "garythung-trashnet")
TFRECORD_DIRECTORY = os.path.join(CNN_DIRECTORY, "..", "data", "tfrecords")

TRAIN_FILENAME = "train.tfrecords"
TEST_FILENAME = "test.tfrecords"

CATEGORIES = {
    'cardboard': 0,
    'glass': 1,
    'metal': 2,
    'paper': 3,
    'plastic': 4
}

SPLIT_RATIO = 0.3

def get_images_and_labels(file_dir, categories):
    images = []
    subfolders = []
    for root, sub_folders, files in os.walk(file_dir):
        for name in files:
            category = root.replace("\\", "/").split('/')[-1]
            if "jpg" in name and category in categories:
                images.append(os.path.join(root, name))
        for name in sub_folders:
            if name in categories:
                subfolders.append(os.path.join(root, name))

    labels = []
    for subfolder in subfolders:        
        no_images = len(os.listdir(subfolder))
        category = subfolder.replace("\\", "/").split('/')[-1]
        labels = np.append(labels, [categories[category]] * no_images)

    temp = np.array([images, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]

    return image_list, label_list

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

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
    filename = os.path.join(save_dir, name)
    writer = tf.python_io.TFRecordWriter(filename)
    for i in np.arange(0, len(labels)):
        try:
            path = images[i].replace("\\", "/").split('/')[-2:]
            print("Converting image: {0}/{1}".format(path[0],path[1]))
            image = tf.gfile.FastGFile(images[i], 'rb').read()
            label = int(labels[i])
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': bytes_feature(image),
                'label': int64_feature(label)
            }))
            writer.write(example.SerializeToString())
            print("Converted image: {0}/{1}".format(path[0],path[1]))
        except IOError as e:
            print("Exception encountered: {0}".format(e))
            continue
    writer.close()

def main(argv):
    images, labels = get_images_and_labels(IMAGE_DIRECTORY, CATEGORIES)
    train_images, train_labels, test_images, test_labels = split(images, labels, SPLIT_RATIO)
    convert_to_tfrecord(train_images, train_labels, TFRECORD_DIRECTORY, TRAIN_FILENAME)
    convert_to_tfrecord(test_images, test_labels, TFRECORD_DIRECTORY, TEST_FILENAME)

if __name__ == "__main__":
    tf.app.run()