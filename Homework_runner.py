import sys

from HW09_Sasuri_Ruzan import HW09_Sasuri_Ruzan_knn as knn
from HW09_Sasuri_Ruzan import HW_09_Sasuri_Ruzan_Trainer as trainer
from HW09_Sasuri_Ruzan import HW_09_Sasuri_Ruzan_Classifier_Train as classifier
from HW09_Sasuri_Ruzan import HW_09_Sasuri_Ruzan_Accuracy_Calculator as accuracy


def main(file_name, clean_file):
	knn.main(file_name)
	print('kNN done.')
	trainer.main(clean_file)
	print('Training done.')
	classifier.main(clean_file)
	print('Classification done.')
	accuracy.main(clean_file)


if __name__ == '__main__':
	main(sys.argv[1], sys.argv[2])
