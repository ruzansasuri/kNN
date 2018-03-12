
import sys

def main(file_name):
	training_file = open(file_name, "r")
	classified_file = open("HW_09_Sasuri_Ruzan_MyClassifications.csv", "w")
	training_file.readline()
	classified_file.write("Class\n")

	for point in training_file:
		values = point.strip().split(",")
		for i in range(len(values)):
			values[i] = float(values[i])
		value_class = -1
		if values[4] <= 2.8699999999999983:
			value_class = 1
		else:
			value_class = 0

		classified_file.write(str(value_class) + "\n")
		print(value_class)

if __name__ == "__main__":
	main(sys.argv[1])