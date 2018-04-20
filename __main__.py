import sys

from svm import train_svm_sift_kmeans, train_svm_orb_kmeans, train_svm_rgb_sift_kmeans, \
    train_svm_inception_bottleneck, train_svm_rgb_gray_sift_kmeans, train_skopt_svm_sift_kmeans_params

def main(training_mode):
    ({
        'svm_inception_bottleneck': train_svm_inception_bottleneck,
        'svm_sift_kmeans': train_svm_sift_kmeans,
        'svm_orb_kmeans': train_svm_orb_kmeans,
        'svm_rgb_sift_kmeans': train_svm_rgb_sift_kmeans,
        'svm_rgb_gray_sift_kmeans': train_svm_rgb_gray_sift_kmeans,
        'skopt_svm_sift_kmeans_params': train_skopt_svm_sift_kmeans_params
    })[training_mode]()

'''
    Usage: . <training_mode>
    training_mode: "svm_inception_bottleneck", "svm_sift_kmeans", "svm_orb_kmeans", 
        "svm_rgb_sift_kmeans", "svm_rgb_gray_sift_kmeans", "skopt_svm_sift_kmeans_params"
'''
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: . <training_mode>')
        sys.exit(1)
    training_mode = sys.argv[1]
    main(training_mode)