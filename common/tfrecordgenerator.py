from matplotlib.image import imread
import tensorflow as tf
import sys
import os

IMAGE_COUNT_PER_CLASS = int(os.environ.get('IMAGE_COUNT_PER_CLASS', 0))
DATASET_PATH = os.environ.get('DATASET_PATH',
    os.path.dirname(__file__) + '/../data/garythung-trashnet/')

path_tfrecords_train = os.path.join(DATASET_PATH, "../tfrecords/train.tfrecords")
# path_tfrecords_train

train_path = [
    os.path.join(DATASET_PATH, "cardboard"),
    os.path.join(DATASET_PATH, "glass"),
    os.path.join(DATASET_PATH, "metal"),
    # os.path.join(DATASET_PATH, "paper"),
    # os.path.join(DATASET_PATH, "plastic"),
    # os.path.join(DATASET_PATH, "trash")
]
def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def print_progress(count, total):
    # Percentage completion.
    pct_complete = float(count) / total

    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


def convert(image_paths, labels, out_path):
    # Args:
    # image_paths   List of file-paths for the images.
    # labels        Class-labels for the images.
    # out_path      File-path for the TFRecords output file.
    onlyfiles = []
    print("Converting: " + out_path)
    for pathi in image_paths :
        print(pathi)
        os.listdir(pathi)
        onlyfiles = onlyfiles + [os.path.join(pathi, f) for f in os.listdir(pathi) if f.endswith('.jpg') and os.path.isfile(os.path.join(pathi, f))]
    # onlyfiles =
    # Number of images. Used when printing the progress.
    num_images = len(onlyfiles)
    # print(onlyfiles[0])
    # print(num_images)

    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(out_path) as writer:
        # Iterate over all the image-paths and class-labels.
        for i, (path) in enumerate(onlyfiles):
            # Print the percentage-progress.
            print_progress(count=i, total=num_images - 1)
            print(path)

            # Load the image-file using matplotlib's imread function.
            img = imread(path)
            # Convert the image to raw bytes.
            img_bytes = img.tostring()

            # Create a dict with the data we want to save in the
            # TFRecords file. You can add more relevant data here.
            data = \
                {
                    'image': wrap_bytes(img_bytes),
                    'label': wrap_int64(labels)
                }

            # Wrap the data as TensorFlow Features.
            feature = tf.train.Features(feature=data)

            # Wrap again as a TensorFlow Example.
            example = tf.train.Example(features=feature)

            # Serialize the data.
            serialized = example.SerializeToString()

            # Write the serialized data to the TFRecords file.
            writer.write(serialized)

convert(image_paths=train_path, # path to the directory of one of the classes
        labels=0, # class id Cardboard maps to 0 and so on
        out_path=path_tfrecords_train)