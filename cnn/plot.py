import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm,
                          target_names):

	cmap = plt.get_cmap('Blues')

	plt.figure(figsize=(5, 5))
	plt.imshow(cm, interpolation='nearest', cmap=cmap)

	if target_names is not None:
		tick_marks = np.arange(len(target_names))
		plt.xticks(tick_marks, target_names)
		plt.yticks(tick_marks, target_names)

	thresh = cm.max() / 2
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, "{:,}".format(cm[i, j]),
					horizontalalignment="center",
					color="white" if cm[i, j] > thresh else "black")
	
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.tight_layout()
	plt.show()

def main():
	# op_zfnet_cm = np.array([
	# 	[83, 11, 2, 20, 5],
	# 	[16, 72, 6, 24, 19],
	# 	[14, 32, 12, 45, 16],
	# 	[18, 7, 9, 157, 5],
	# 	[24, 22, 4, 25, 69]])
	# labels = ['Carboard', 'Glass', 'Metal', 'Paper', 'Plastic']
	# plot_confusion_matrix(op_zfnet_cm, labels)

	svm_cm = np.array([
		[83, 18, 9, 22, 11],
		[10, 74, 16, 3, 27],
		[4, 26, 69, 3, 13],
		[9, 12, 4, 69, 5],
		[8, 21, 12, 9, 63]])
	labels = ['Paper', 'Glass', 'Plastic', 'Metal', 'Cardboard']
	plot_confusion_matrix(svm_cm, labels)

if __name__ == "__main__":
	main()