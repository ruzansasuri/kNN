"""
Authors: Ruzan Sasuri
Nov 10th, 2017.
"""
import sys

CLASS = ['Greyhound', 'Whippet']


def main(file_name):
	file_in = open(file_name, 'r')
	file_out = open('HW_09_Sasuri_Ruzan_MyClassifications_train.csv', 'r')

	file_in.readline()
	file_out.readline()

	TP = 0
	TN = 0
	total = 0

	for line in file_in:
		class1 = line.strip().split(',')[-1]
		class2 = file_out.readline().strip()
		total += 1
		if class1 == '1' and class2 == '1':
			TP += 1
		if class1 == '0' and class2 == '0':
			TN += 1

	print(round((TP + TN) / total * 100, 3), '%')

if __name__ == '__main__':
	main(sys.argv[1])
