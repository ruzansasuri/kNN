"""
Authors: Ruzan Sasuri
Nov 10th, 2017.
"""

import sys


class Neighbor:
	"""
	Class that stores a neighbor's point and distance to the point.
	"""
	__slots__ = 'point', 'distance'

	def __init__(self, point, dist):
		"""
		Creates a new Neighbor object.
		:param point: The neighbor's index.
		:param dist: The distnace fron the neighbor to the point.
		"""
		self.point = point
		self.distance = dist


class Point:
	"""
	Stores a point and its nearest neighbors.
	"""
	__slots__ = 'k', 'point', 'neighbors'

	def __init__(self, point, k):
		"""
		Creates a new Point object.
		:param point: The point's index in the data set.
		:param k: The k value to be used.
		"""
		self.k = k
		self.point = point
		self.neighbors = []

	def try_neighbor(self, neighbor):
		"""
		Tries if a point can be added as this point's neighbor.
		:param neighbor: The point to try for.
		:return: None
		"""
		if len(self.neighbors) < self.k:
			self.neighbors.append(neighbor)
		elif neighbor.distance < self.neighbors[0].distance:
			self.neighbors[0] = neighbor
		self.neighbors = sorted(self.neighbors, key=lambda nei: nei.distance)

	def find_class(self, classes):
		"""
		Find's the mode of the point's nearest neighbor's classes and returns it.
		:param classes: The list of classes of all the records.
		:return: The new class value.
		"""
		class_0_count = 0
		class_1_count = 0
		if len(self.neighbors) < self.k:
			return -1
		for neighbor in self.neighbors:
			if neighbor.point >= len(classes):
				print(neighbor.point)
			if classes[neighbor.point] == 1:
				class_1_count += 1
			else:
				class_0_count += 1
		if class_0_count > class_1_count:
			return 0
		elif class_0_count < class_1_count:
			return 1
		else:
			return classes[self.point]


def file_check(file, permission='r'):
	"""
    Creates a file handler.
    :param file: Name of t`he file
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
    :return: list of attributes, list of data points and list of classes.
    """
	attrs = file.readline().strip().split(',')
	for i in range(len(attrs)):
		attrs[i] = attrs[i].strip()
	data_points = []
	classes = []
	for line in file:
		if line == '\n' or line.strip() == '':
			continue
		line = line.strip().split(',')
		point = []
		if line[-1] == 'Greyhound':
			line[-1] = 1
		else:
			line[-1] = 0
		for ind in range(len(line) - 1):
			point.append(float(line[ind]))
		data_points.append(point)
		classes.append(int(line[-1]))
	return attrs, data_points, classes


def find_square_euc(point1, point2):
	"""
	Finds the distance from one point to another as the square of the eulidean distance.
	:param point1: The first point
	:param point2: The second point.
	:return: The square of the euclidean distance.
	"""
	dist = 0
	for a in range(len(point1)):
		diff = point1[a] - point2[a]
		dist += pow(diff, 2)
	return dist


def find_nn(data_points, k, threshold=-1):
	"""
	Finds and stores the nearest neighbors to all the points.
	:param data_points: The data.
	:param k: The value of k for k-NN.
	:param threshold: The distance threshold out of which we don't use the points which don't have k neighbors.
					  Default is -1.
	:return: A list of Point objects.
	"""
	points = []
	for i in range(len(data_points)):
		points.append(Point(i, k))
		for j in range(len(data_points)):
			if i == j:
				continue
			dist = find_square_euc(data_points[i], data_points[j])
			points[i].try_neighbor(Neighbor(j, dist))
	if threshold != -1:
		del_list = []
		for i in range(len(points)):
			for neighbor in points[i].neighbors:
				if neighbor.distance > threshold:
					del_list.append(i)
					break
		point_list = []
		for i in range(len(points)):
			if i not in del_list:
				point_list.append(points[i])
				# new_classes.append(classes[i])
		points = point_list
	return points


def find_new_classes(points, classes):
	"""
	Finds the new classes of each of the remaining points as the mode of the class of the nearest neighbors.
	:param points: The points that remain.
	:param classes: The class values.
	:return: The list of new classes and their miss-classification count.
	"""
	new_classes = []
	miss_class = 0
	for i in range(len(points)):
		c = points[i].find_class(classes)
		if c == -1:
			print(i)
		new_classes.append(c)
		if new_classes[i] != classes[i]:
			miss_class += 1
	return new_classes, miss_class


def for_a_k_knn(data_points, classes, k):
	"""
	Runs k-NN for a single point.
	:param data_points: The data.
	:param classes: The classes.
	:param k: The k value.
	:return: The list of the new classes.
	"""
	initial_classes = [c for c in classes]
	points, unused = find_nn(data_points, k)
	y_list1 = []
	y_list2 = []
	for _ in range(10):
		classes, miss_class = find_new_classes(points, classes)
		y_list2.append(miss_class)
		miss_class = 0
		for i in range(len(initial_classes)):
			if initial_classes[i] != classes[i]:
				miss_class += 1
		y_list1.append(miss_class)
	return classes


def knn(data_points, classes, k, threshold=-1):
	"""
	Finds the nearest neighbors and changes the classes by calling their functions.
	It then finds the miss-classification rate.
	:param data_points: The data.
	:param classes: The classes
	:param k: The k value.
	:param threshold: The distance threshold. Default is -1.
	:return: The new classes and the miss-classification rate.
	"""
	points = find_nn(data_points, k, threshold=threshold)
	new_classes, miss_class = find_new_classes(points, classes)
	miss_class = 0
	for i in range(len(points)):
		if classes[points[i].point] != new_classes[i]:
			miss_class += 1
	return classes, miss_class / len(data_points)


def for_multiple_k(data_points, classes, threshold=-1):
	"""
	Runs k-NN for multiple values of k, specifically even values of k from 1 to the number of records - 1.
	:param data_points: The data.
	:param classes: The classes.
	:param threshold: The distance threshold. Default is -1.
	:return: The best k value and its miss-classification rate.
	"""
	y_list = []
	best_missclass = sys.maxsize
	best_k = 0
	for k in range(1, len(data_points), 2):
		print(k)
		new_classes, miss_class = knn(data_points, classes, k, threshold=threshold)
		if miss_class < best_missclass:
			best_missclass = miss_class
			best_k = k
		y_list.append(miss_class)
	return best_k, best_missclass


def write_line(line, file):
	"""
	Writes a line to the csv file.
	:param line: The line to write.
	:param file: The file to write to.
	:return: None
	"""
	for i in range(len(line) - 1):
		file.write(str(line[i]) + ',')
	file.write(str(line[-1]) + '\n')


def write_to_file(attr, data_points, points, classes, file):
	"""
	Writes the points and the new classes to a csv file.
	:param attr: The list of attributes.
	:param data_points: The data.
	:param points: The list of the remaining point object.
	:param classes: The classes.
	:param file: The file to write to.
	:return: None
	"""
	write_line(attr, file)
	zero = 0
	one = 0
	for i in range(len(points)):
		point = points[i].point
		if classes[point] == 1:
			one += 1
			class_val = 'Greyhound'
		else:
			zero += 1
			class_val = 'Whippet'
		write_line(data_points[point] + [class_val], file)
	print(zero, one)


def main(file_name):
	file_in = file_check(file_name)
	attr, data_points, classes = read_csv(file_in)
	k, miss_class = for_multiple_k(data_points, classes, threshold=8)
	print(k)
	# k = 3
	classes, miss_class = knn(data_points, classes, k=k)
	file_out = file_check('CleanSet.csv', 'w')
	points = find_nn(data_points, k=1, threshold=8)
	write_to_file(attr, data_points, points, classes, file_out)

if __name__ == '__main__':
	main(sys.argv[1])
