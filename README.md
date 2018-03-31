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

## Running SVM Classifier

```bash
python3 ./svm/svm_classifier.py
```

## Running CNN Classifier

```bash
python3 ./cnn/cnn_classifier.py
```

## CNN Architecture

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

1. Input Layer (MNIST data)

    Output Shape: `[-1, 384, 512, 1]`