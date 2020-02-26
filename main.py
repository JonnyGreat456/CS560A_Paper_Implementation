#!/usr/bin/env python3

import irisloader
import svm
import numpy as np
import plotly.graph_objects as go

TRAINING_TEST_RATIO = 0.8

if __name__ == "__main__":
	X_raw, Y_raw = irisloader.load_data()
	num_features, num_samples = X_raw.shape

	# Randomly shuffle data
	rand_indices = np.random.permutation(num_samples)
	X = X_raw[:, rand_indices]
	Y = Y_raw[:, rand_indices]

	# Split into training and test sets
	split_point = int(num_samples * TRAINING_TEST_RATIO)
	X_training = X[:, 0:split_point]
	Y_training = Y[:, 0:split_point]

	X_testing = X[:, split_point:]
	Y_testing = Y[:, split_point:]

	# Create and train the classifer
	classifier = svm.SVMPegasos(num_features=X.shape[0])
	objvals = classifier.fit(X_training, Y_training, T=50, k=X_training.shape[1])

	fig = go.Figure()
	fig.add_trace(go.Scatter(x=list(range(1, X_training.shape[1]+1)), y=objvals[1:], mode='lines+markers'))
	fig.update_layout(title='Loss vs Iterations', xaxis_title='Iteration', yaxis_title='Loss')
	fig.show()

	# Test accuracy
	predictions = classifier.predict(X_testing)
	correct = 0
	total = X_testing.shape[1]
	for i in range(predictions.shape[1]):
	 	yp = predictions[0,i]
	 	ytrue = Y_testing[0,i]
	 	if yp*ytrue > 0:
	 		correct += 1

	print("Accuracy: {} correct out of {} ({}%)".format(correct, total, correct / total * 100))