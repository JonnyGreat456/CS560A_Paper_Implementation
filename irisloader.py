#!/usr/bin/env python3

file_name = "./datasets/iris/iris.data"
import numpy as np

def load_data():
	X = []
	Y = []

	with open(file_name) as f:
		for line in f:
			if line.strip() == "":
				continue
			s = line.split(",")
			data = [float(x) for x in s[0:4]]
			X.append(data)
			if s[4].strip() == "Iris-virginica":
				Y.append(1)
			else:
				Y.append(-1)
	return (np.asmatrix(X).T, np.asmatrix(Y))