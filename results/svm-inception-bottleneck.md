# Results for training a SVM using Inception V3 bottleneck features

## Settings

```
IMAGE_COUNT_PER_CLASS=400
```

Using cardboard, glass, metal, paper, plastic image datasets from [trashnet](https://github.com/garythung/trashnet)

Using pre-trained [Inception V3 image classification model](https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1) to compute feature vectors of images.

## Output Log

```
Training model for fold 0...
Testing model for fold 0...
Accuracy: 0.85
Confusion matrix:
[[72  0  1  4  0]
 [ 0 61  3  0 10]
 [ 0 13 73  2  7]
 [ 8  1  1 68  0]
 [ 1  5  4  0 66]]
Training model for fold 1...
Testing model for fold 1...
Accuracy: 0.8725
Confusion matrix:
[[58  0  0  2  0]
 [ 0 67  2  0  4]
 [ 1  4 79  5  7]
 [11  0  3 81  0]
 [ 1  8  2  1 64]]
Training model for fold 2...
Testing model for fold 2...
Accuracy: 0.83
Confusion matrix:
[[69  0  0  2  0]
 [ 0 59  7  0  8]
 [ 1 10 64  1  8]
 [10  2  5 77  0]
 [ 0 11  3  0 63]]
Training model for fold 3...
Testing model for fold 3...
Accuracy: 0.845
Confusion matrix:
[[69  0  0  4  0]
 [ 0 65  5  0  3]
 [ 1 13 71  2 11]
 [10  0  0 77  2]
 [ 0  7  3  1 56]]
Training model for fold 4...
Testing model for fold 4...
Accuracy: 0.8525
Confusion matrix:
[[76  0  0  1  0]
 [ 0 66  7  1 11]
 [ 1  3 61  1  8]
 [11  2  1 69  3]
 [ 0  3  5  1 69]]



Combined accuracy: 0.85
Combined confusion matrix:
[[344   0   1  13   0]
 [  0 318  24   1  36]
 [  4  43 348  11  41]
 [ 50   5  10 372   5]
 [  2  34  17   3 318]]
 ```