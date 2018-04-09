import sys

from svm import train_svm_raw_pixels, train_svm_inception_bottleneck, train_svm_sift_kmeans, train_svm_orb_kmeans, train_svm_colour_orb_kmeans

'''
    Usage: . <training_mode>
    training_mode: "svm_raw_pixels", "svm_inception_bottleneck", "svm_sift_kmeans", "svm_orb_kmeans", "svm_colour_orb_kmeans"
'''
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: . <training_mode>')
        sys.exit(1)

    training_mode = sys.argv[1]
    ({
        'svm_raw_pixels': train_svm_raw_pixels,
        'svm_inception_bottleneck': train_svm_inception_bottleneck,
        'svm_sift_kmeans': train_svm_sift_kmeans,
        'svm_orb_kmeans': train_svm_orb_kmeans,
        'svm_colour_orb_kmeans': train_svm_colour_orb_kmeans
    })[training_mode]()