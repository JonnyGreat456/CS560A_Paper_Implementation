#!/usr/bin/env python3

import numpy as np

file_name = "./datasets/ccat/Index_EN-EN"
vocabulary_size = 21531
sample_size = 18758

def load_data():
	X = np.zeros((vocabulary_size, sample_size))
	Y = []

	class1 = 0
	class2 = 0
	sample_num = 0
	with open(file_name) as f:
		for line in f:
			if line.strip() == "":
				continue
			s = line.strip().split(" ")
			label = s[0]
			s = s[1:]

			for feature in s:
				contents = feature.split(":")
				feature_num = int(contents[0]) - 1
				X[feature_num,sample_num] = contents[1]

			if label == "CCAT":
				Y.append(1)
				class1 += 1
			else:
				Y.append(-1)
				class2 += 1
			sample_num += 1
	return (X, np.asmatrix(Y), class1, class2)