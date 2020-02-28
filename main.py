#!/usr/bin/env python3

import ccatloader
import uspsloader
import svm
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import fetch_rcv1
import time

TRAINING_TEST_RATIO = 0.8

def get_accuracy(classifier, X, Y):
	predictions = classifier.predict(X)
	correct = 0
	total = X.shape[1]
	cmatrix = np.zeros((2,2))
	for i in range(predictions.shape[1]):
	 	yp = predictions[0,i]
	 	if yp == 0:
	 		yp == 1
	 	ytrue = Y[0,i]
	 	if yp*ytrue > 0:
	 		correct += 1

	 	cmatrix[int(max(0,ytrue)), int(max(0,yp))] += 1
	return (correct, total, cmatrix)

def run_linear_ccat():
	print("[Running linear CCAT]")
	print("Loading data...", end='', flush=True)
	X, Y, numpos, numneg = ccatloader.load_data()
	num_features, num_samples = X.shape
	print("Loaded {} samples with {} features each ({} samples in the positive class, {} samples in the negative class).".format(num_samples, num_features, numpos, numneg))

	# Split into training and test sets. The CCAT dataset has a custom split
	print("Dividing data into training and test sets...", end='', flush=True)
	X_testing = X[:, 0:23149]
	Y_testing = Y[:, 0:23149]

	X_training = X[:, 23149:]
	Y_training = Y[:, 23149:]
	print("done")

	# Create and train the classifer
	getObjVals = False
	print("Training the classifier...", end='', flush=True)
	classifier = svm.PegasosLinear(num_features=num_features, lam=0.00018)
	start_time = time.time()
	objvals = classifier.fit(X_training, Y_training, T=10000, k=1, getObjVals=getObjVals)
	elapsed_time = time.time() - start_time
	print("trained in {}s".format(elapsed_time))

	# Plot
	if getObjVals:
		print("Plotting...", end='', flush=True)
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=list(range(1, X_training.shape[1]+1)), y=objvals[1:], mode='lines+markers'))
		fig.update_layout(title='Loss vs Iterations', xaxis_title='Iteration', yaxis_title='Loss')
		fig.show()
		print("done")

	print("Calculating accuracy:")
	# Training accuracy
	correct, total, cmatrix = get_accuracy(classifier, X_training, Y_training)
	print("Training Set Accuracy: {} correct out of {} ({}%)".format(correct, total, correct / total * 100))
	print("Confusion Matrix:\n{}".format(cmatrix))

	# # Test accuracy
	correct, total, cmatrix = get_accuracy(classifier, X_testing, Y_testing)
	print("Testing Set Accuracy: {} correct out of {} ({}%)".format(correct, total, correct / total * 100))
	print("Confusion Matrix:\n{}".format(cmatrix))

def run_gaussian_usps():
	print("Loading data...", end='', flush=True)
	X_training, Y_training, X_testing, Y_testing = uspsloader.load_data()
	print("Loaded {} training samples and {} testing samples with {} features each.".format(X_training.shape[1], X_testing.shape[1], X_training.shape[0]))

	print("Converting labels...", end='', flush=True)
	pos = 0
	neg = 0
	for i in range(Y_training.shape[1]):
		if Y_training[0, i] == 8:
			Y_training[0, i] = 1
			pos += 1
		else:
			Y_training[0, i] = -1
			neg += 1

	for i in range(Y_testing.shape[1]):
		if Y_testing[0, i] == 8:
			Y_testing[0, i] = 1
		else:
			Y_testing[0, i] = -1

	print("{} total positive samples and {} total negative samples".format(pos, neg))

	getObjVals = False
	print("Training the classifier...", end='', flush=True)
	classifier = svm.PegasosGaussian(lam=0.000136, gamma=2.0)
	start_time = time.time()
	objvals = classifier.fit(X_training, Y_training, T=30000, getObjVals=getObjVals)
	elapsed_time = time.time() - start_time
	print("trained in {}s".format(elapsed_time))

	# Plot
	if getObjVals:
		print("Plotting...", end='', flush=True)
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=list(range(1, X_training.shape[1]+1)), y=objvals[1:], mode='lines+markers'))
		fig.update_layout(title='Loss vs Iterations', xaxis_title='Iteration', yaxis_title='Loss')
		fig.show()
		print("done")

	print("Calculating accuracy:")
	# Training accuracy
	correct, total, cmatrix = get_accuracy(classifier, X_training, Y_training)
	print("Training Set Accuracy: {} correct out of {} ({}%)".format(correct, total, correct / total * 100))
	print("Confusion Matrix:\n{}".format(cmatrix))

	# Test accuracy
	correct, total, cmatrix = get_accuracy(classifier, X_testing, Y_testing)
	print("Testing Set Accuracy: {} correct out of {} ({}%)".format(correct, total, correct / total * 100))
	print("Confusion Matrix:\n{}".format(cmatrix))

if __name__ == "__main__":
	# run_linear_ccat()
	run_gaussian_usps()


