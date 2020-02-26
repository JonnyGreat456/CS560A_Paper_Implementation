#!/usr/bin/env python3

import numpy as np

class SVMPegasos:
	w = None
	b = None
	lam = 0.0

	def __init__(self, num_features, lam=0.25):
		self.w = np.zeros((num_features, 1))
		self.b = np.zeros((num_features, 1))
		self.lam = lam

	def loss(self, X, Y):
		wX = X.transpose().dot(self.w).transpose()
		omywX = 1.0 - np.multiply(wX, Y) 
		zeroArr = np.zeros(omywX.shape)
		hinge = np.maximum(omywX, zeroArr)
		avgHinge = np.mean(hinge)
		return avgHinge + (self.lam / 2.0 * float(self.w.T.dot(self.w)))

	# k = 1 -> SGD
	# k = number of samples -> GD
	def fit(self, X, Y, T, k=1, getObjVals=True):
		objvals = [0.0] * (T+1)
		if getObjVals:
			objvals[0] = self.loss(X, Y)

		# Generate all randoms at once for efficiency
		num_features, num_samples = X.shape
		indices = np.random.randint(0, num_samples, size=k*T)
		indices_i = 0

		# Train
		for t in range(1, T+1):
			# Set eta = 1/λt
			eta = 1.0 / (self.lam * t)

			# Calculate subgradients
			update = np.zeros((num_features, 1))
			for ci in range(k):
				i = indices[indices_i + ci]
				xi = X.getcol(i)
				yi = Y[:, i].item()
				if yi * xi.transpose().dot(self.w).item() < 1:
					update += xi.multiply(yi) 
			update /= k
			indices_i += k

			# Update w
			self.w = np.multiply(self.w, (1.0 - 1.0 / t)) + eta * update

			if getObjVals:
				objvals[t] = self.loss(X, Y)
		return objvals

	def predict(self, X):
		predictions = X.T.dot(self.w).T
		return np.sign(predictions)

