#!/usr/bin/env python3

import numpy as np
from numpy import linalg as LA

class PegasosLinear:
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

class PegasosGaussian:
	a = None
	X = None
	Y = None
	T = 1
	lam = 0.0
	gamma = 1.0

	def __init__(self, lam=0.25, gamma=1.0):
		self.lam = lam
		self.gamma = gamma

	def kernel(self, xi, xj):
		return np.exp(-1.0 * self.gamma * (np.square(LA.norm(xi - xj, ord=2, axis=0))))

	def loss(self, X, Y, t):
		reg2 = 0.0
		hinge = np.zeros(Y.shape)
		for i in range(X.shape[1]):
			xi = X[:, i].reshape((X.shape[0], 1))
			yi = Y[:, i].item()
			ai = self.a[0, i]

			kn = self.kernel(X, xi)
			aprime = ai * self.a
			yprime = yi * Y
			aykn = np.multiply(aprime, np.multiply(yprime, kn))
			reg2 += ((1 / (self.lam * t)) ** 2) * np.sum(aykn)

			xikn = (1 / (self.lam * t)) * np.multiply(self.a, np.multiply(Y, kn))
			xiknhinge = max(0.0, 1.0 - yi * np.sum(xikn))
			hinge[0, i] = xiknhinge
		reg2 *= (self.lam / 2.0)
		avgHinge = np.mean(hinge)
		return reg2 + avgHinge

	def fit(self, X, Y, T, getObjVals=True):
		objvals = [0.0] * (T+1)

		# Generate all randoms at once for efficiency
		num_features, num_samples = X.shape
		self.a = np.zeros((1, num_samples))
		self.X = X
		self.Y = Y
		self.T = T
		indices = np.random.randint(0, num_samples, size=T)

		# Train
		for t in range(1, T+1):
			# Set eta = 1/λt
			eta = 1.0 / (self.lam * t)

			i = indices[t-1]
			xi = X[:, i].reshape((X.shape[0], 1))
			yi = Y[:, i].item()

			kn = self.kernel(X, xi)
			aykn = np.multiply(self.a, yi * kn)
			s = np.sum(aykn)

			if yi * eta * s < 1:
				self.a[0, i] += 1

			if getObjVals:
				objvals[t] = self.loss(X, Y, t)
		return objvals

	def predict(self, X):
		predictions = np.zeros((1, X.shape[1]))
		for i in range(X.shape[1]):
			xi = X[:, i].reshape((X.shape[0], 1))

			kn = self.kernel(self.X, xi)
			ykn = np.multiply(self.Y, kn)
			aykn = np.multiply(self.a, ykn)
			s = np.sum(aykn)

			if (1.0 / (self.lam * self.T)) * s >= 0:
				predictions[0, i] = 1
			else:
				predictions[0, i] = -1

		return predictions