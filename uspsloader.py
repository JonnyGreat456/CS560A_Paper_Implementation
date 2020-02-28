#!/usr/bin/env python3

import h5py
import numpy as np

path = "./datasets/usps/usps.h5"

def load_data():
	with h5py.File(path, 'r') as hf:
		train = hf.get('train')
		X_tr = train.get('data')[:]
		y_tr = train.get('target')[:]
		test = hf.get('test')
		X_te = test.get('data')[:]
		y_te = test.get('target')[:]

		sub = X_te[0:38, :]
		suby = y_te[0:38]
		X_tr = np.vstack((X_tr, sub))
		y_tr = np.append(y_tr, suby)

		X_te = X_te[38:, :]
		y_te = y_te[38:]

		return (X_tr.T, y_tr.reshape(1, len(y_tr)), X_te.T, y_te.reshape(1, len(y_te)))