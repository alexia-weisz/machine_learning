import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as optimize
import random
from pdb import set_trace


"""
 Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

  Instructions
  ------------

  This file contains code that helps you get started on the
  linear exercise. You will need to complete the following functions
  in this exericse:

     lrCostFunction.m (logistic regression cost function)
     oneVsAll.m
     predictOneVsAll.m
     predict.m

  For this exercise, you will not need to change any code in this file,
  or any other files other than those mentioned above.

"""

def display_data(X, example_width=None):
	"""
	display_data Display 2D data in a nice grid
	   h, display_array = display_data(X, example_width) displays 2D data
	   stored in X in a nice grid. It returns the figure handle h and the 
	   displayed array if requested.
	"""
	# Set example_width automatically if not passed in
	if example_width is None or example_width.size == 0:
		example_width = int(np.round(np.sqrt(X.shape[1])))
	
	# Gray Image
	cmap = plt.cm.gray

	# Compute rows, cols
	m, n = X.shape
	example_height = n / example_width

	# Compute number of items to display
	display_rows = int(np.floor(np.sqrt(m)))
	display_cols = int(np.ceil(m / display_rows))

	# Between images padding
	pad = 1

	# Setup blank display
	display_array = -np.ones((pad + display_rows * (example_height + pad), 
							  pad + display_cols * (example_width + pad)))

	# Copy each example into a patch on the display array
	curr_ex = 0

	for j in range(1, display_rows + 1):
		for i in range(1, display_cols + 1):
			if curr_ex > m:
				break
			# Copy the patch
			
			# Get the max value of the patch
			try:
				max_val = np.max(np.abs(X[curr_ex, :]))
			except IndexError:
				set_trace()
			#set_trace()
			x_inds = pad + (j - 1) * (example_height + pad) + np.arange(1, example_height+1)
			y_inds = pad + (i - 1) * (example_width + pad) + np.arange(1, example_width+1)
			new_val = X[curr_ex, :].reshape(example_height, example_width) / max_val
			#set_trace()
			display_array[x_inds[0]:x_inds[-1]+1, y_inds[0]: y_inds[-1]+1] = new_val
			curr_ex += 1
		if curr_ex > m: 
			break

	# Display Image
	h = plt.imshow(display_array.T, origin='upper', cmap=cmap, vmin=-1, vmax=1)

	# Do not show axis
	plt.gca().axes.get_yaxis().set_visible(False)
	plt.gca().axes.get_xaxis().set_visible(False)

	plt.show()

	return h, display_array

def sigmoid(z):
	"""
	Hypothesis for Logistic Regression
	"""
	return 1. / (1 +  np.exp(-z))

def lr_cost_function(theta, X, y, llambda):
	"""
	LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
	regularization
	   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
	   theta as the parameter for regularized logistic regression and the
	   gradient of the cost w.r.t. to the parameters. 
	"""
	if len(theta.shape) == 1:
		theta = theta.reshape(theta.shape[0], 1)
	
	if len(y.shape) == 1:
		y = y.reshape(y.shape[0], 1)

	m = len(y) # number of training examples
	h = sigmoid(np.dot(X, theta))

	theta_reg = np.append(0, theta[1:])
	theta_reg = theta_reg.reshape(theta_reg.shape[0], 1)

	j1 = np.dot(y.T, np.log(h))
	j2 = np.dot((1-y).T, np.log(1-h))
	reg = (llambda / 2.) * np.dot(theta_reg.T, theta_reg)

	J = (-1. / m) * (j1 + j2 - reg)
	grad = (1. / m) * (np.dot(X.T, (h-y)) + llambda * theta_reg)

	return J[0,0], grad[:,0]

def lr_cost(theta, X, y, llambda):
	"""
	LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
	regularization
	   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
	   theta as the parameter for regularized logistic regression and the
	   gradient of the cost w.r.t. to the parameters. 
	"""
	if len(theta.shape) == 1:
		theta = theta.reshape(theta.shape[0], 1)
	
	if len(y.shape) == 1:
		y = y.reshape(y.shape[0], 1)

	m = len(y) # number of training examples
	h = sigmoid(np.dot(X, theta))

	theta_reg = np.append(0, theta[1:])
	theta_reg = theta_reg.reshape(theta_reg.shape[0], 1)

	j1 = np.dot(y.T, np.log(h))
	j2 = np.dot((1-y).T, np.log(1-h))
	reg = (llambda / 2.) * np.dot(theta_reg.T, theta_reg)

	J = (-1. / m) * (j1 + j2 - reg)

	return J[0,0]

def lr_grad(theta, X, y, llambda):
	"""
	LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
	regularization
	   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
	   theta as the parameter for regularized logistic regression and the
	   gradient of the cost w.r.t. to the parameters. 
	"""
	if len(theta.shape) == 1:
		theta = theta.reshape(theta.shape[0], 1)
	
	if len(y.shape) == 1:
		y = y.reshape(y.shape[0], 1)

	m = len(y) # number of training examples
	h = sigmoid(np.dot(X, theta))

	theta_reg = np.append(0, theta[1:])
	theta_reg = theta_reg.reshape(theta_reg.shape[0], 1)

	grad = (1. / m) * (np.dot(X.T, (h-y)) + llambda * theta_reg)

	return grad[:,0]

def one_vs_all(X, y, num_labels, llambda):
	"""
	function [all_theta] = oneVsAll(X, y, num_labels, lambda)
	 ONEVSALL trains multiple logistic regression classifiers and returns all
	 the classifiers in a matrix all_theta, where the i-th row of all_theta 
	 corresponds to the classifier for label i
	    [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
	    logistic regression classifiers and returns each of these classifiers
	    in a matrix all_theta, where the i-th row of all_theta corresponds 
	    to the classifier for label i
	"""
	# Some useful variables
	m, n = X.shape

	# You need to return the following variables correctly 
	all_theta = np.zeros((num_labels, n + 1))

	# Add ones to the X data matrix
	X = np.append(np.ones((m, 1)), X, axis=1)

	for c in range(0, num_labels):
		initial_theta = np.zeros((n+1, 1))

		# get the correct classificiton percentage with this one -- slow
		#kwargs = {'maxiter': 50, 'args': (X, (y == c+1), llambda)}
		#theta = optimize.fmin_ncg(f=lr_cost, x0=initial_theta, fprime=lr_grad, **kwargs)

		# but not with this one, which is much faster -- the eigth iteration is funky
		#kwargs = {'maxiter': 50, 'args': (X, (y==(c+1)), llambda)}
		#theta = optimize.fmin_cg(f=lr_cost, x0=initial_theta, fprime=lr_grad, **kwargs)

		# This works with method=BFGS, Netwon-CG, but not quite with CG.
		kwargs = {'args': (X, (y==(c+1)), llambda), 'method': 'BFGS'}
		theta = optimize.minimize(fun=lr_cost, x0=initial_theta, jac=lr_grad, **kwargs)['x']

		all_theta[c, :] = theta
	
	return all_theta

def predict_one_vs_all(all_theta, X):
	"""
	function p = predictOneVsAll(all_theta, X)
	PREDICT Predict the label for a trained one-vs-all classifier. The labels 
	are in the range 1..K, where K = size(all_theta, 1). 
	  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
	  for each example in the matrix X. Note that X contains the examples in
	  rows. all_theta is a matrix where the i-th row is a trained logistic
	  regression theta vector for the i-th class. You should set p to a vector
	  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
	  for 4 examples) 
	"""
	m = X.shape[0]
	num_labels = all_theta.shape[0]

	# You need to return the following variables correctly 
	p = np.zeros((m, 1))

	# Add ones to the X data matrix
	X = np.append(np.ones((m, 1)), X, axis=1)

	h = sigmoid(np.dot(X, all_theta.T))
	p = np.argmax(h, axis=1) + 1

	return p[:,None]

def predict(Theta1, Theta2, X):
	"""
	function p = predict(Theta1, Theta2, X)
	PREDICT Predict the label of an input given a trained neural network
	   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
	   trained weights of a neural network (Theta1, Theta2)
	"""

	m = X.shape[0]
	num_labels = Theta2.shape[0]

	# You need to return the following variables correctly 
	p = np.zeros((m, 1))

	# Level 1
	X = np.append(np.ones((m, 1)), X, axis=1)
	z2 = np.dot(Theta1, X.T)
	a2 = sigmoid(z2)

	#Level 2 -- add ones
	mm = a2.shape[1]
	a2 = np.append(np.ones((m, 1)), a2.T, axis=1)
	z3 = np.dot(Theta2, a2.T)

	# Level 3
	h = sigmoid(z3)

	p = np.argmax(h, axis=0) + 1

	return p[:,None]


def multi_class_classification():
	# Setup the parameters you will use for this part of the exercise
	input_layer_size  = 400  # 20x20 Input Images of Digits
	num_labels = 10          # 10 labels, from 1 to 10
	                         # (note that we have mapped "0" to label 10)

	## =========== Part 1: Loading and Visualizing Data =============
	#  We start the exercise by first loading and visualizing the dataset.
	#  You will be working with a dataset that contains handwritten digits.
	#

	# Load Training Data
	print('Loading and Visualizing Data ...\n')

	data = loadmat('ex3data1.mat') # training data stored in arrays X, y
	X, y = data['X'], data['y']
	m = X.shape[0]

	# Randomly select 100 data points to display
	rand_indices = random.sample(range(1, m+1), m)
	sel = X[rand_indices[0:100], :]

	h, display_array = display_data(sel)

	raw_input('Program paused. Press enter to continue.\n')


	## ============ Part 2a: Vectorize Logistic Regression ============
	#  In this part of the exercise, you will reuse your logistic regression
	#  code from the last exercise. You task here is to make sure that your
	#  regularized logistic regression implementation is vectorized. After
	#  that, you will implement one-vs-all classification for the handwritten
	#  digit dataset.


	# Test case for lr_cost_function
	print('Testing lr_cost_function() with regularization')

	theta_t = np.asarray([-2, -1, 1, 2])[:, None]
	arr1 = np.ones((5, 1))
	arr2 = np.asarray(range(1, 16)).reshape(5, 3, order='F')/10.
	X_t = np.append(arr1, arr2, axis=1)
	y_t = (np.asarray([1, 0, 1, 0, 1])[:, None] >= 0.5)
	lambda_t = 3
	J, grad = lr_cost_function(theta_t, X_t, y_t, lambda_t)

	print('Cost: {:f}'.format(J))
	print('Expected cost: 2.534819')
	print('Gradients:\n')
	print(' {}'.format(grad))
	print('Expected gradients:\n')
	print(' 0.146561 -0.548558 0.724722 1.398003')

	raw_input('Program paused. Press enter to continue.\n')


	## ============ Part 2b: One-vs-All Training ============
	print('\nTraining One-vs-All Logistic Regression...\n')

	llambda = 0.1
	all_theta = one_vs_all(X, y, num_labels, llambda)

	raw_input('Program paused. Press enter to continue.\n')


	## ================ Part 3: Predict for One-Vs-All ================

	pred = predict_one_vs_all(all_theta, X)
	print('Training Set Accuracy: {:f}'.format(np.mean((pred == y)) * 100))

def neural_networks():

	# Setup the parameters you will use for this exercise
	input_layer_size  = 400  # 20x20 Input Images of Digits
	hidden_layer_size = 25   #25 hidden units
	num_labels = 10          # 10 labels, from 1 to 10   
	                         # (note that we have mapped "0" to label 10)

	## =========== Part 1: Loading and Visualizing Data =============
	#  We start the exercise by first loading and visualizing the dataset. 
	#  You will be working with a dataset that contains handwritten digits.
	#

	# Load Training Data
	print('Loading and Visualizing Data ...\n')

	data = loadmat('ex3data1.mat')
	X, y = data['X'], data['y']
	m = X.shape[0]

	# Randomly select 100 data points to display
	rand_indices = random.sample(range(1, m+1), m)
	sel = X[rand_indices[0:100], :]

	h, display_array = display_data(sel)

	raw_input('Program paused. Press enter to continue.\n')



	# ================ Part 2: Loading Pameters ================
	# In this part of the exercise, we load some pre-initialized 
	# neural network parameters.

	print('\nLoading Saved Neural Network Parameters ...\n')

	# Load the weights into variables Theta1 and Theta2
	data = loadmat('ex3weights.mat')
	Theta1, Theta2 = data['Theta1'], data['Theta2']



	# ================= Part 3: Implement Predict =================
	#  After training the neural network, we would like to use it to predict
	#  the labels. You will now implement the "predict" function to use the
	#  neural network to predict the labels of the training set. This lets
	#  you compute the training set accuracy.

	pred = predict(Theta1, Theta2, X)
	print('\nTraining Set Accuracy: {:f}'.format(np.mean(pred == y) * 100))

	raw_input('Program paused. Press enter to continue.\n')

	#  To give you an idea of the network's output, you can also run
	#  through the examples one at the a time to see what it is predicting.

	#  Randomly permute examples
	rp = np.asarray(random.sample(range(0, m), m))

	for i in range(0, m):
		print('\nDisplaying Example Image\n')
	    dx = X[rp[i], :]
	    dx = dx.reshape(1, dx.shape[0])

	    h, da = display_data(dx)

	    pred = predict(Theta1, Theta2, dx)
	    print('\nNeural Network Prediction: {} (digit {})\n'.format(pred, np.mod(pred, 10)))
	    
	    # Pause with quit option
	    s = raw_input('Paused - press enter to continue, q to exit')
	    if s == 'q':
	    	break



if __name__ == '__main__':
	multi_class_classification()
	neural_networks()


