# Results for training a SVM using Inception V3 bottleneck features

Using cardboard, glass, metal, paper, plastic image datasets from [trashnet](https://github.com/garythung/trashnet)

Using pre-trained [Inception V3 image classification model](https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1) to compute feature vectors of images.

## Settings & Results

```
IMAGE_COUNT_PER_CLASS=400
```
```
Combined accuracy: 0.85
Combined confusion matrix:
[[344   0   1  13   0]
 [  0 318  24   1  36]
 [  4  43 348  11  41]
 [ 50   5  10 372   5]
 [  2  34  17   3 318]]
 ```

```
IMAGE_COUNT_PER_CLASS=100
```
```
Combined train accuracy: 0.8715
Combined train confusion matrix:
[[339   0   0   5   3]
 [  0 318  16   4  35]
 [  2  45 373   7  28]
 [ 59   4  10 384   5]
 [  0  33   1   0 329]]
Combined test accuracy: 0.784
Combined test confusion matrix:
[[74  0  0  4  1]
 [ 1 69  8  1 14]
 [ 2 16 88  4 12]
 [23  2  3 91  3]
 [ 0 13  1  0 70]]
```