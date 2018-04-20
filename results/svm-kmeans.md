# Results for training SVM using SIFT/ORB features + KMeans

## Settings & Results for SIFT features + KMeans

Run with `python . svm_sift_kmeans`.

(1) 

```
IMAGE_COUNT_PER_CLASS=100
KMEANS_CLUSTERS=128
```
```
Combined accuracy: 0.266
Combined confusion matrix:
[[10  3  0  3  0]
 [ 8 18  4  7 13]
 [60 54 77 65 67]
 [15 19 16 21 13]
 [ 7  6  3  4  7]]
```

(2) Binary Bag-of-Features SIFT-KMean features

```
IMAGE_COUNT_PER_CLASS=100
KMEANS_CLUSTERS=128
IF_BINARY_FEATURES=True
```
```
Combined accuracy: 0.306
Combined confusion matrix:
[[49 32 17 29 23]
 [ 7 10 10  4  3]
 [ 9 23 24  9 19]
 [25 15 16 28 13]
 [10 20 33 30 42]]
```

(3) Binary Bag-of-Features SIFT-KMean features with number of kmeans clusters = sqrt(total keypoints)

```
IMAGE_COUNT_PER_CLASS=50
IF_SQRT_KEYPOINTS_KMEANS_CLUSTERS=True
IF_BINARY_FEATURES=True
```
```
Combined accuracy: 0.284
Combined confusion matrix:
[[18 18  8 12  5]
 [20 14 18 16 10]
 [ 3  8  5  0  5]
 [ 4  1  9 14 10]
 [ 5  9 10  8 20]]
 ```

(4) Removed binary features, add normalization

```
SVM_C_PARAM=1
IMAGE_COUNT_PER_CLASS=100
IF_SQRT_KEYPOINTS_KMEANS_CLUSTERS=True
IF_NORMALIZE_IMAGES=True
```
```
Combined train accuracy: 0.927
Combined train confusion matrix:
[[360   4   0   7   5]
 [ 40 390  14  23  46]
 [  0   3 386   0   0]
 [  0   0   0 369   0]
 [  0   3   0   1 349]]
Combined test accuracy: 0.446
Combined test confusion matrix:
[[38  7  2  6  5]
 [23 46 18 19 25]
 [ 4 11 34  1  8]
 [32 20 39 71 28]
 [ 3 16  7  3 34]]
 ```

(5) Added binary features

```
SVM_C_PARAM=1
IMAGE_COUNT_PER_CLASS=50
IF_SQRT_KEYPOINTS_KMEANS_CLUSTERS=True
IF_NORMALIZE_IMAGES=True
IF_BINARY_FEATURES=True
```
```
Combined train accuracy: 0.561
Combined train confusion matrix:
[[119  29  22  47  19]
 [ 44 128  45  32  20]
 [  3   3  55   2   0]
 [ 12   9  31 108  10]
 [ 22  31  47  11 151]]
Combined test accuracy: 0.288
Combined test confusion matrix:
[[14 18  7 11  5]
 [22 14 18 16 11]
 [ 1  3  2  0  2]
 [ 6  2  9 16  6]
 [ 7 13 14  7 26]]
 ```

(6) 

```
SVM_C_PARAM=1
IMAGE_COUNT_PER_CLASS=100
IF_SQRT_KEYPOINTS_KMEANS_CLUSTERS=True
IF_NORMALIZE_IMAGES=True
IF_BINARY_FEATURES=True
```
```
Combined train accuracy: 0.653
Combined train confusion matrix:
[[335  93  84 118  77]
 [  8 251  36  15  15]
 [  7  10 194   1   7]
 [ 27  14  40 247  22]
 [ 23  32  46  19 279]]
Combined test accuracy: 0.4
Combined test confusion matrix:
[[68 39 24 36 24]
 [ 9 27 25  6  7]
 [ 2  4 12  4 10]
 [14  9 18 45 11]
 [ 7 21 21  9 48]]
 ```

## Settings & Results for ORB features + KMeans

Run with `python . svm_orb_kmeans`.

(1) Binary Bag-of-Features ORB-KMean features with number of kmeans clusters = sqrt(total keypoints)
```
IMAGE_COUNT_PER_CLASS=50
IF_SQRT_KEYPOINTS_KMEANS_CLUSTERS=True
IF_BINARY_FEATURES=True
```
```
Combined accuracy: 0.328
Combined confusion matrix:
[[18 12  0 11  8]
 [ 5  7  5  1  3]
 [ 3 12 25  8 21]
 [18 10  9 24 10]
 [ 6  9 11  6  8]]
 ```

(2) 
```
IMAGE_COUNT_PER_CLASS=100
IF_SQRT_KEYPOINTS_KMEANS_CLUSTERS=True
IF_BINARY_FEATURES=True
```
```
Combined accuracy: 0.372
Combined confusion matrix:
[[45 32 12 17 22]
 [ 7 11  9  4  1]
 [11 29 50 18 39]
 [31 11 14 56 14]
 [ 6 17 15  5 24]
 ```

(3) 
```
IMAGE_COUNT_PER_CLASS=20
IF_SQRT_KEYPOINTS_KMEANS_CLUSTERS=True
IF_BINARY_FEATURES=True
```
```
Combined train accuracy: 0.42
Combined train confusion matrix:
[[30  3  0  2  7]
 [ 0 17  1  0  0]
 [13 22 41 24 20]
 [27 17 25 47 20]
 [10 21 13  7 33]]
Combined test accuracy: 0.13
Combined test confusion matrix:
[[ 3  0  1  2  4]
 [ 1  0  0  0  0]
 [ 7  6  5 12  4]
 [ 4  8 11  3 10]
 [ 5  6  3  3  2]]
 ```

(4) Increased SVM's C param
```
IMAGE_COUNT_PER_CLASS=20
SVM_C_PARAM=1000
IF_SQRT_KEYPOINTS_KMEANS_CLUSTERS=True
IF_BINARY_FEATURES=True
```
```
Combined train accuracy: 1.0
Combined train confusion matrix:
[[80  0  0  0  0]
 [ 0 80  0  0  0]
 [ 0  0 80  0  0]
 [ 0  0  0 80  0]
 [ 0  0  0  0 80]]
Combined test accuracy: 0.26
Combined test confusion matrix:
[[5 2 0 6 4]
 [2 6 3 1 1]
 [2 4 6 6 5]
 [5 3 7 4 5]
 [6 5 4 3 5]]
 ```

(5) Increased SVM's C param

```
IMAGE_COUNT_PER_CLASS=100
SVM_C_PARAM=1000
IF_SQRT_KEYPOINTS_KMEANS_CLUSTERS=True
IF_BINARY_FEATURES=True
```
```
Combined train accuracy: 0.9985
Combined train confusion matrix:
[[400   0   0   0   0]
 [  0 398   0   0   1]
 [  0   0 400   0   0]
 [  0   0   0 400   0]
 [  0   2   0   0 399]]
Combined test accuracy: 0.344
Combined test confusion matrix:
[[41 20 17 22 14]
 [10 30 21 14 22]
 [14 24 33 16 20]
 [23  6  8 39 15]
 [12 20 21  9 29]]
 ```

(4) Decreased SVM's C param

```
IMAGE_COUNT_PER_CLASS=50
SVM_C_PARAM=10
IF_SQRT_KEYPOINTS_KMEANS_CLUSTERS=True
IF_BINARY_FEATURES=True
```
```
Combined train accuracy: 0.98
Combined train confusion matrix:
[[200   4   0   7   4]
 [  0 195   0   0   1]
 [  0   0 200   0   0]
 [  0   1   0 193   3]
 [  0   0   0   0 192]]
Combined test accuracy: 0.352
Combined test confusion matrix:
[[23  8  7 16  7]
 [10 16 11  4 15]
 [ 2 10 19 10  9]
 [13  7  5 15  4]
 [ 2  9  8  5 15]]
 ```

(5) Normalized images

```
IMAGE_COUNT_PER_CLASS=100
SVM_C_PARAM=1
IF_NORMALIZE_IMAGES=True
IF_SQRT_KEYPOINTS_KMEANS_CLUSTERS=True
IF_BINARY_FEATURES=True
```
```
Combined train accuracy: 0.6655
Combined train confusion matrix:
[[258  52  22  42  53]
 [  8 214   9   2  10]
 [ 38  66 305  38  43]
 [ 80  24  33 298  38]
 [ 16  44  31  20 256]]
Combined test accuracy: 0.372
Combined test confusion matrix:
[[40 24 10 16 16]
 [ 8 22  9  6  7]
 [11 27 44 18 33]
 [33  9 14 52 16]
 [ 8 18 23  8 28]]
 ```

## Settings & Results for RGB-ORB features + KMeans

Run with `python . svm_colour_orb_kmeans`.

(1)
```
IMAGE_COUNT_PER_CLASS=20
IF_SQRT_KEYPOINTS_KMEANS_CLUSTERS=True
IF_BINARY_FEATURES=True
```
```
Combined train accuracy: 0.4575
Combined train confusion matrix:
[[30  2  0  4  7]
 [ 0 18  1  0  1]
 [11 16 47 16 12]
 [24 16 19 48 20]
 [15 28 13 12 40]]
Combined test accuracy: 0.15
Combined test confusion matrix:
[[ 4  0  0  1  3]
 [ 0  0  0  0  1]
 [ 5  4  5  9  3]
 [ 5  7 11  3 10]
 [ 6  9  4  7  3]]
 ```

(2) Increased SVM's C param
```
IMAGE_COUNT_PER_CLASS=20
SVM_C_PARAM=1000
IF_SQRT_KEYPOINTS_KMEANS_CLUSTERS=True
IF_BINARY_FEATURES=True
```
```
Combined train accuracy: 1.0
Combined train confusion matrix:
[[80  0  0  0  0]
 [ 0 80  0  0  0]
 [ 0  0 80  0  0]
 [ 0  0  0 80  0]
 [ 0  0  0  0 80]]
Combined test accuracy: 0.26
Combined test confusion matrix:
[[5 2 1 9 3]
 [3 6 4 1 2]
 [3 2 8 5 9]
 [6 1 4 3 2]
 [3 9 3 2 4]]
 ```

(3) Decreased SVM's C param
```
IMAGE_COUNT_PER_CLASS=20
SVM_C_PARAM=100
IF_SQRT_KEYPOINTS_KMEANS_CLUSTERS=True
IF_BINARY_FEATURES=True
```
```
Combined train accuracy: 1.0
Combined train confusion matrix:
[[80  0  0  0  0]
 [ 0 80  0  0  0]
 [ 0  0 80  0  0]
 [ 0  0  0 80  0]
 [ 0  0  0  0 80]]
Combined test accuracy: 0.28
Combined test confusion matrix:
[[ 6  4  2  5  5]
 [ 2  7  4  0  0]
 [ 3  5  5  7 11]
 [ 7  0  4  6  0]
 [ 2  4  5  2  4]]
 ```

(4) Decreased SVM's C param
```
IMAGE_COUNT_PER_CLASS=20
SVM_C_PARAM=10
IF_SQRT_KEYPOINTS_KMEANS_CLUSTERS=True
IF_BINARY_FEATURES=True
```
```
Combined train accuracy: 1.0
Combined train confusion matrix:
[[80  0  0  0  0]
 [ 0 80  0  0  0]
 [ 0  0 80  0  0]
 [ 0  0  0 80  0]
 [ 0  0  0  0 80]]
Combined test accuracy: 0.26
Combined test confusion matrix:
[[5 4 1 6 3]
 [1 5 2 0 3]
 [5 2 5 6 5]
 [5 1 7 5 3]
 [4 8 5 3 6]]
 ```

(5)
```
IMAGE_COUNT_PER_CLASS=50
SVM_C_PARAM=10
IF_SQRT_KEYPOINTS_KMEANS_CLUSTERS=True
IF_BINARY_FEATURES=True
```
```
Combined train accuracy: 0.983
Combined train confusion matrix:
[[200   4   0   4   5]
 [  0 195   0   0   1]
 [  0   0 200   0   0]
 [  0   1   0 196   2]
 [  0   0   0   0 192]]
Combined test accuracy: 0.412
Combined test confusion matrix:
[[26 11  5 16  5]
 [ 3 14  5  4 10]
 [ 5  9 22  6  8]
 [12  7  6 20  6]
 [ 4  9 12  4 21]]
 ```
 
## Settings & Results for RGB-SIFT features + KMeans

Run with `python . svm_colour_sift_kmeans`.

(1) 

```
SVM_C_PARAM=10
IMAGE_COUNT_PER_CLASS=50
IF_SQRT_KEYPOINTS_KMEANS_CLUSTERS=True
IF_BINARY_FEATURES=True
```
```
Combined train accuracy: 0.979
Combined train confusion matrix:
[[198   0   0   8   6]
 [  2 200   0   2   3]
 [  0   0 200   0   0]
 [  0   0   0 190   0]
 [  0   0   0   0 191]]
Combined test accuracy: 0.484
Combined test confusion matrix:
[[27 12  3 14  8]
 [ 6 17 11  4  6]
 [ 4 18 24  1  6]
 [10  1  7 29  6]
 [ 3  2  5  2 24]]
```
