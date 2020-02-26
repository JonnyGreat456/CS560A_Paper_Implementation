#!/usr/bin/env python3

import numpy as np
from sklearn.datasets import fetch_rcv1

def load_data():
	rcv1 = fetch_rcv1()
	X = rcv1.data.T
	num_samples = X.shape[1]

	# Find the index for 'CCAT'
	ccat_index = -1
	for i, label in enumerate(rcv1.target_names):
		if label == 'CCAT':
			ccat_index = i
			break

	# Convert encoding to {-1, 1}
	Y = np.zeros((1, num_samples))

	numpos = 0
	numneg = 0
	for i in range(rcv1.target.shape[0]):
		y = rcv1.target[i, ccat_index]
		if y == 1:
			numpos += 1
			Y[0, i] = 1
		else:
			numneg += 1
			Y[0, i] = -1

	return (X.tocsc(), Y, numpos, numneg)