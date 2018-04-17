# Trash Classifier

## Setup

```bash
cd ~
mkdir .envs
cd .envs
mkdir thrash-classifier
virtualenv --no-site-packages -p python3 ./thrash-classifier
cd #project-directory
source ~/.envs/thrash-classifier/bin/activate
pip3 install -r requirements.txt
```

## Adding Python Modules via Pip

```bash
pip3 install #new-python-module
pip3 freeze > requirements.txt
cat requirements.txt
git push
```

## Converting Images to TFRecords

```bash
python3 ./cnn/image_to_tfrecord.py
```

## Running SVM Classifier

```bash
python3 ./svm/svm_classifier.py
```

## Running CNN Classifier

```bash
python3 ./cnn/cnn_classifier.py -a #alexnet/zfnet
```

## Dataset

We obtained our dataset of 2527 trash images from [garythung/trashnet](https://github.com/garythung/trashnet). The pictures were taken by placing the object on a white posterboard and using sunlight and/or room lighting. They have been resized down to `256 x 256` when converting the images to tfrecords. The devices used to take the photos were Apple iPhone 7 Plus, Apple iPhone 5S, and Apple iPhone SE.

## CNN Architecture (AlexNet)

Methods in `layers` module expect input tensors to have shape `[batch_size, image_height, image_width, channels]`.

Under `'SAME'` padding scheme, output is calulated as such:
```python
out_height = ceil(float(in_height) / float(strides[1]))
out_width  = ceil(float(in_width) / float(strides[2]))
```

Under `'VALID'` padding scheme, output is calulated as such:
```python
out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
out_width  = ceil(float(in_width - filter_width + 1) / float(strides[2]))
```

1. Input Layer (garythung dataset)

    Output Shape: `[-1, 256, 256, 3]`

2. Convolution Layer 1

    Input Shape: `[-1, 256, 256, 1]`  
    Filter Shape: `[11, 11]`  
    Number of Filters: `96`  
    Strides Shape: `[4, 4]`  
    Output Shape (Same Padding): `[-1, 64, 64, 96]`  
    Activation Function: `ReLU`

3. Pooling Layer 1

    Input Shape: `[-1, 64, 64, 96]`  
    Filter Shape: `[3, 3]`  
    Strides Shape: `[2, 2]`  
    Output Shape (Valid Padding): `[-1, 31, 31, 96]`

4. Convolution Layer 2

    Input Shape: `[-1, 31, 31, 1]`  
    Filter Shape: `[5, 5]`  
    Number of Filters: `192`  
    Strides Shape: `[1, 1]`  
    Output Shape (Same Padding): `[-1, 31, 31, 192]`  
    Activation Function: `ReLU`

5. Pooling Layer 2

    Input Shape: `[-1, 31, 31, 192]`  
    Filter Shape: `[3, 3]`  
    Strides Shape: `[2, 2]`  
    Output Shape (Valid Padding): `[-1, 15, 15, 192]`

6. Convolution Layer 3

    Input Shape: `[-1, 15, 15, 192]`  
    Filter Shape: `[3, 3]`  
    Number of Filters: `288`  
    Strides Shape: `[1, 1]`  
    Output Shape (Same Padding): `[-1, 15, 15, 288]`  
    Activation Function: `ReLU`

7. Convolution Layer 4

    Input Shape: `[-1, 15, 15, 288]` 
    Filter Shape: `[3, 3]`  
    Number of Filters: `288`  
    Strides Shape: `[1, 1]`  
    Output Shape (Same Padding): `[-1, 15, 15, 288]`  
    Activation Function: `ReLU`

8. Convolution Layer 5

    Input Shape: `[-1, 15, 15, 288]` 
    Filter Shape: `[3, 3]`  
    Number of Filters: `192`  
    Strides Shape: `[1, 1]`  
    Output Shape (Same Padding): `[-1, 15, 15, 192]`  
    Activation Function: `ReLU`

9. Pooling Layer 3

    Input Shape: `[-1, 15, 15, 192]`  
    Filter Shape: `[3, 3]`  
    Strides Shape: `[2, 2]`  
    Output Shape (Valid Padding): `[-1, 7, 7, 192]`

10. Dense Layer 1

    Input Shape: `[-1, 7 * 7 * 192]`  
    Number of Neurons: `4096`  
    Output Shape: `[-1, 4096]`  
    Activation Function: `ReLU`  

11. Dropout Layer 1

    Input Shape: `[-1, 4096]`  
    Dropout Rate: `0.2`  
    Output Shape: `[-1, 4096]`  

12. Dense Layer 2

    Input Shape: `[-1, 4096]`  
    Number of Neurons: `4096`  
    Output Shape: `[-1, 4096]`  
    Activation Function: `ReLU`  

13. Dropout Layer 2

    Input Shape: `[-1, 4096]`  
    Dropout Rate: `0.2`  
    Output Shape: `[-1, 4096]`  

14. Logits Layer

    Input Shape: `[-1, 4096]`  
    Number of Neurons: `5`  
    Output Shape: `[-1, 5]`  

## CNN Architecture (ZfNet)

Methods in `layers` module expect input tensors to have shape `[batch_size, image_height, image_width, channels]`.

Under `'SAME'` padding scheme, output is calulated as such:
```python
out_height = ceil(float(in_height) / float(strides[1]))
out_width  = ceil(float(in_width) / float(strides[2]))
```

Under `'VALID'` padding scheme, output is calulated as such:
```python
out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
out_width  = ceil(float(in_width - filter_width + 1) / float(strides[2]))
```

1. Input Layer (garythung dataset)

    Output Shape: `[-1, 256, 256, 3]`

2. Convolution Layer 1

    Input Shape: `[-1, 256, 256, 1]`  
    Filter Shape: `[7, 7]`  
    Number of Filters: `96`  
    Strides Shape: `[2, 2]`  
    Output Shape (Same Padding): `[-1, 128, 128, 96]`  
    Activation Function: `ReLU`

3. Pooling Layer 1

    Input Shape: `[-1, 128, 128, 96]`  
    Filter Shape: `[3, 3]`  
    Strides Shape: `[2, 2]`  
    Output Shape (Valid Padding): `[-1, 63, 63, 96]`