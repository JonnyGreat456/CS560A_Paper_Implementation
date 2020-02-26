#!/usr/bin/env python3

import irisloader
import ccatloader
import svm
import numpy as np
import plotly.graph_objects as go

TRAINING_TEST_RATIO = 0.8

def get_accuracy(classifier, X, Y):
	predictions = classifier.predict(X)
	correct = 0
	total = X.shape[1]
	for i in range(predictions.shape[1]):
	 	yp = predictions[0,i]
	 	ytrue = Y[0,i]
	 	if yp*ytrue > 0:
	 		correct += 1
	return (correct, total)

if __name__ == "__main__":
	print("Loading data...", end='', flush=True)
	X_raw, Y_raw, numpos, numneg = ccatloader.load_data()
	num_features, num_samples = X_raw.shape
	print("Loaded {} samples with {} features each ({} samples in the positive class, {} samples in the negative class).".format(num_samples, num_features, numpos, numneg))

	# Randomly shuffle data
	print("Shuffling data...", end='', flush=True)
	rand_indices = np.random.permutation(num_samples)
	X = X_raw[:, rand_indices]
	Y = Y_raw[:, rand_indices]
	print("done")

	# Split into training and test sets
	print("Dividing data into training and set sets...", end='', flush=True)
	split_point = int(num_samples * TRAINING_TEST_RATIO)
	X_training = X[:, 0:split_point]
	Y_training = Y[:, 0:split_point]

	X_testing = X[:, split_point:]
	Y_testing = Y[:, split_point:]
	print("done")

	# Create and train the classifer
	print("Training the classifier...", end='', flush=True)
	classifier = svm.SVMPegasos(num_features=num_features, lam=0.0001)
	objvals = classifier.fit(X_training, Y_training, T=50, k=1)
	print("done")

	# Plot
	print("Plotting...", end='', flush=True)
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=list(range(1, X_training.shape[1]+1)), y=objvals[1:], mode='lines+markers'))
	fig.update_layout(title='Loss vs Iterations', xaxis_title='Iteration', yaxis_title='Loss')
	fig.show()
	print("done")

	print("Calculating accuracy:")
	# Training accuracy
	correct, total = get_accuracy(classifier, X_training, Y_training)
	print("Training Set Accuracy: {} correct out of {} ({}%)".format(correct, total, correct / total * 100))

	# Test accuracy
	correct, total = get_accuracy(classifier, X_testing, Y_testing)
	print("Testing Set Accuracy: {} correct out of {} ({}%)".format(correct, total, correct / total * 100))