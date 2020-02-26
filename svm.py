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

	def hinge(self, x, y):
		return max(0.0, 1 - y * float(self.w.T.dot(x)))

	def loss(self, X, Y):
		l = 0.0
		for i in range(X.shape[1]):
			x = X[:, i]
			y = float(Y[:, i])
			l += self.hinge(x, y)
		l /= X.shape[1]
		l += self.lam / 2.0 * float(self.w.T.dot(self.w))
		return l

	# k = 1 -> SGD
	# k = number of samples -> GD
	def fit(self, X, Y, T, k=1):
		objvals = [0.0] * (T+1)
		objvals[0] = self.loss(X, Y)

		num_features, num_samples = X.shape
		for t in range(1, T+1):
			# Randomly choose indices
			indices = np.random.choice(num_samples, k, replace=False)

			# Set eta = 1/λt
			eta = 1.0 / (self.lam * t)

			# Calculate subgradients
			update = np.zeros((num_features, 1))
			for i in indices:
				xi = X[:, i]
				yi = float(Y[:, i])
				if yi * self.w.T.dot(xi) < 1:
					update += yi * xi
			update /= k

			# Update w
			self.w = self.w * (1.0 - 1.0 / t) + eta * update
			objvals[t] = self.loss(X, Y)
		return objvals

	def predict(self, X):
		predictions = self.w.T.dot(X)
		return np.sign(predictions)

