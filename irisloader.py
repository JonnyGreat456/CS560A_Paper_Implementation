#!/usr/bin/env python3

import numpy as np

file_name = "./datasets/iris/iris.data"

def load_data():
	X = []
	Y = []

	class1 = 0
	class2 = 0
	with open(file_name) as f:
		for line in f:
			if line.strip() == "":
				continue
			s = line.split(",")
			data = [float(x) for x in s[0:4]]
			X.append(data)
			if s[4].strip() == "Iris-virginica":
				Y.append(1)
				class1 += 1
			else:
				Y.append(-1)
				class2 += 1
	return (np.asmatrix(X).T, np.asmatrix(Y), class1, class2)