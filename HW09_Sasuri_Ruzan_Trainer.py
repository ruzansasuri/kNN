"""
Authors: Ruzan Sasuri
Oct 22nd, 2017.
"""
import sys
from HW09_Sasuri_Ruzan import HW09_Sasuri_Ruzan_Classifier_Train as classifier
from HW09_Sasuri_Ruzan import HW09_Sasuri_Ruzan_Accuracy_Calculator as accuracy

CLASS = []


class DecisionNode:
	"""
	Contains the node's decision info.
	"""
	__slots__ = 'attribute', 'threshold', 'left', 'right', 'class_value'

	def __init__(self):
		self.attribute = -1
		self.threshold = 0
		self.left = None
		self.right = None
		self.class_value = -1

	def create_decision(self, attribute, threshold, class_value=-1):
		"""
		Creates a decision node.
		:param attribute: The attribute to split on.
		:param threshold: The threshold to split at.
		:param class_value: Class to be assigned to the left split.
		:return:
		"""
		self.attribute = attribute
		self.threshold = threshold
		self.left = DecisionNode()
		self.right = DecisionNode()
		self.class_value = class_value

	def make_leaf(self, class_value):
		"""
		Makes the current node a leaf node.
		:param class_value: Class to be assigned to the left split.
		:return:
		"""
		self.attribute = -1
		self.threshold = 0
		self.class_value = class_value

	def make_children_leaves(self, class_value):
		"""
		Makes both children the leaves.
		:param class_value: Class to be assigned to the left split.
		:return:
		"""
		self.left = DecisionNode()
		self.right = DecisionNode()
		self.left.class_value = class_value
		self.right.class_value = abs(class_value - 1)


class DecisionTree:
	"""
	Stores the decision tree
	"""
	__slots__ = 'root'

	def __init__(self):
		self.root = DecisionNode()

	def dfs(self, indent):
		"""
		Calls the helper _dfs() to convert the tree to a code string.
		:param indent: The indentation for the root.
		:return: The code string
		"""
		return self._dfs(self.root, indent)

	def _dfs(self, node, indent):
		"""
		Recursively traverses through the tree and converts a node to a string.
		:param node: The current node.
		:param indent: The indentation for this node.
		:return: The code string.
		"""
		code_string = ''
		for i in range(indent):
			code_string += '\t'
			print('\t', end='')
		if node.attribute == -1:
			code_string += 'value_class = ' + str(node.class_value) + '\n'
			print('value_class =', node.class_value)
			return code_string
		code_string += 'if values[' + str(node.attribute) + '] <= ' + str(node.threshold) + ':\n'
		print('if values[' + str(node.attribute) + '] <= ' + str(node.threshold) + ':')
		code_string += self._dfs(node.left, indent + 1)
		for i in range(indent):
			code_string += '\t'
			print('\t', end='')
		code_string += 'else:\n'
		print('else:')
		code_string += self._dfs(node.right, indent + 1)
		return code_string


def file_check(file, permission):
	"""
    Creates a file handler.
    :param file: Name of the file
    :param permission: Permission
    :return: File handler
    """
	try:
		f = open(file, permission)
		return f
	except FileNotFoundError:
		print("File", file, "does not exist...")
		exit()


def read_csv(file):
	"""
    Reads the csv file and converts it into data points and attriutes.
    :param file: File handler
    :return: list of attributes, list of data points and list of attributes to ignore.
    """
	attrs = file.readline().strip().split(',')[1:]
	data_points = []
	for line in file:
		if line == '\n' or line.strip() == '':
			continue
		line = line.strip().split(',')
		point = []
		for ind in range(len(line) - 1):
			point.append(float(line[ind]))
		if line[-1] not in CLASS:
			CLASS.append(line[-1])
		point.append(CLASS.index(line[-1]))
		data_points.append(point)
	return attrs, data_points


def find_thresholds(index, data_points):
	"""
	Finds the threshold values for an attribute.
	:param index: The attribute's index.
	:param data_points: The data.
	:return: A list of threshold values for this attribute.
	"""
	min_val = min(data_points, key=lambda point: point[index])[index]
	max_val = max(data_points, key=lambda point: point[index])[index]

	thresholds = []
	curr_threshold = min_val
	while curr_threshold < max_val:
		thresholds.append(curr_threshold)
		curr_threshold += 0.05
	return thresholds


def calculate_purity(attribute_ind, threshold, data_points):
	"""
	Calculates the purity as a weighted GINI for a split candidate.
	:param attribute_ind: The attribute to split on's index.
	:param threshold: The threshold to split on's index.
	:param data_points: The data.
	:return: The wieghted GINI of the split candidate.
	"""
	left_count_zero = 0
	left_count_one = 0
	right_count_zero = 0
	right_count_one = 0
	for point in data_points:
		if point[attribute_ind] <= threshold:
			if point[-1] == 0:
				left_count_zero += 1
			else:
				left_count_one += 1
		else:
			if point[-1] == 0:
				right_count_zero += 1
			else:
				right_count_one += 1
	total_left = left_count_zero + left_count_one
	total_right = right_count_zero + right_count_one
	if total_left == 0 or total_right == 0:
		return float('inf')
	total = total_left + total_right
	gini_left = 1 - pow(left_count_zero / total_left, 2) - pow(left_count_one / total_left, 2)
	gini_right = 1 - pow(right_count_zero / total_right, 2) - pow(right_count_one / total_right, 2)
	weighted_gini = gini_left * total_left / total + gini_right * total_right / total
	return weighted_gini


def split(data_points, attribute, threshold):
	"""
	Splits the data into a left and right segment.
	:param data_points: The data.
	:param attribute: The attribute to split on's index.
	:param threshold: The threshold to split on's index.
	:return: The left and right data segments.
	"""
	left = []
	right = []
	for point in data_points:
		if point[attribute] <= threshold:
			left.append(point)
		else:
			right.append(point)
	return left, right


def find_class(data_points):
	"""
	Finds the mode class for the data.
	:param data_points: The data.
	:return: The class mode.
	"""
	class_zero_count = 0
	class_one_count = 0
	for i in range(len(data_points)):
		if data_points[i][-1] == 0:
			class_zero_count += 1
		else:
			class_one_count += 1
	if class_zero_count > class_one_count:
		return 0
	else:
		return 1


def check_homogenity(data_points):
	"""
	Stopping condition checks if the data is 55% homogeneous.
	:param data_points: The data.
	:return: The class if homogeneous or -1 if not.
	"""
	one = 0
	zero = 0
	for point in data_points:
		if point[-1] == 0:
			zero += 1
		else:
			one += 1
	if zero / len(data_points) >= 0.55:
		return 0
	elif one / len(data_points) >= 0.55:
		return 1
	return -1


def choose_splits(attributes, data_points, current_node):
	"""
	Creates nodes for the decision tree.
	:param attributes: The list of attributes.
	:param data_points: The data.
	:param current_node: The node being looked at.
	:return: None
	"""
	class_value = check_homogenity(data_points)
	if class_value != -1:
		current_node.make_leaf(class_value)
	else:
		best_impurity = float('inf')
		best_attribute = -1
		best_threshold = 0
		for attribute_ind in range(len(attributes) - 1):
			for threshold in find_thresholds(attribute_ind, data_points):
				impurity = calculate_purity(attribute_ind, threshold, data_points)
				if impurity < best_impurity:
					best_impurity = impurity
					best_attribute = attribute_ind
					best_threshold = threshold

		current_node.create_decision(best_attribute, best_threshold)

		left, right = split(data_points, best_attribute, best_threshold)
		if len(left) == 0:
			current_node.make_children_leaves(find_class(right))
		elif len(right) == 0:
			current_node.make_children_leaves(find_class(left))
		else:
			choose_splits(attributes, left, current_node.left)
			choose_splits(attributes, right, current_node.right)
			if current_node.left.attribute == -1 and current_node.right.attribute == -1 and \
							current_node.left.class_value == current_node.right.class_value:
				current_node.make_leaf(current_node.left.class_value)


def write_code(decision_tree, file, training):
	"""
	Writes the code to the file.
	:param decision_tree: The decision tree.
	:param file: The file handler
	:param training: True if training data.
	:return: None
	"""
	code_builder = list()
	code_builder.append('import sys\n'
						'\n'
						'def main(file_name):\n'
						'\ttraining_file = open(file_name, "r")\n')
	if training:
		code_builder.append('\tclassified_file = open("HW_09_Sasuri_Ruzan_MyClassifications_train.csv", "w")\n')
	else:
		code_builder.append('\tclassified_file = open("HW_09_Sasuri_Ruzan_MyClassifications.csv", "w")\n')
	code_builder.append('\ttraining_file.readline()\n'
						'\tclassified_file.write("Class\\n")\n'
						'\n'
						'\tfor point in training_file:\n'
						'\t\tvalues = point.strip().split(",")\n')
	if training:
		code_builder.append('\t\tvalues = values[: len(values) - 1]\n'
							'\t\tfor i in range(len(values) - 1):\n')
	else:
		code_builder.append('\t\tfor i in range(len(values)):\n')
	code_builder.append('\t\t\tvalues[i] = float(values[i])\n'
						'\t\tvalue_class = -1\n')

	code_builder.append(decision_tree.dfs(2))
	code_builder.append('\n'
						'\t\tclassified_file.write(str(value_class) + "\\n")\n'
						'\t\tprint(value_class)\n')
	code_builder.append('\n'
						'if __name__ == "__main__":\n'
						'\tmain(sys.argv[1])')

	for code in code_builder:
		file.write(str(code))


def write_line(line, file):
	for i in range(len(line) - 1):
		file.write(str(line[i]) + ',')
	file.write(str(line[-1]) + '\n')


def write_csv(attributes, data_points, file):
	write_line([0] + attributes, file)
	for point in data_points:
		write_line(point, file)


def main(file_name):
	# file_in = file_check('HW_05C_DecTree_TRAINING__v540.csv', 'r')
	file_in = file_check(file_name, 'r')
	file_out_train = file_check('HW09_Sasuri_Ruzan_Classifier_Train.py', 'w')
	file_out_test = file_check('HW09_Sasuri_Ruzan_Classifier.py', 'w')

	attributes, data_points = read_csv(file_in)

	train_set = data_points[:int(0.7 * len(data_points))]
	test_set = data_points[int(0.7 * len(data_points)):]

	decision_tree = DecisionTree()
	choose_splits(attributes, train_set, decision_tree.root)
	write_code(decision_tree, file_out_train, True)
	write_code(decision_tree, file_out_test, False)

	file_in.close()
	file_out_train.close()
	file_out_test.close()

	file_out = file_check('TestSet.csv', 'w')
	write_csv(attributes, test_set, file_out)
	file_out.close()
	classifier.main('TestSet.csv')
	accuracy.main('TestSet.csv')


if __name__ == '__main__':
	main(sys.argv[1])
