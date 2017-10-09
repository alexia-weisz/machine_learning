import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from pdb import set_trace


def plot_data(x, y, labels=None):
	"""
	Plot data x and y, where x are the data to be plotted and y are the labels
	"""
	admitted = (y == 1)
	not_admitted = (y == 0)

	plt.plot(x[admitted, 0], x[admitted, 1], 'kP', label=labels[0])
	plt.plot(x[not_admitted, 0], x[not_admitted, 1], 'yo', mec='k', mew=0.5, label=labels[1])
	
def sigmoid(z):
	"""
	Hypothesis for Logistic Regression
	"""
	return 1. / (1 +  np.exp(-z))

def cost_function(theta, X, y, J_only=False, grad_only=False):
	"""
	Calculate the cost function for variables theta, matrix X, and training examples y
	
	Parameters
	----------
	theta : array
	   fitting parameters
	X : matrix
	   Variables used in fitting the cost function
	y : array
	   Labels

	Returns
	-------
	J : float
	   cost
	grad : array, same size as theta
	   partial derivative of cost function
	"""
	if len(theta.shape) == 1:
		theta = theta.reshape(theta.shape[0], 1)
	m = len(y) # number of training examples

	h = sigmoid(X.dot(theta))
	J = (1. / m) * np.sum((-y * np.log(h) - (1. - y) * np.log(1. - h)))
	grad = 1. / m * (X.T).dot(h - y)

	if J_only:
		return J
	elif grad_only:
		return grad[:,0]
	else:
		return J, grad[:,0]

def compute_cost(theta, X, y):
	if len(theta.shape) == 1:
		theta = theta.reshape(theta.shape[0], 1)
	m = len(y) # number of training examples
	h = sigmoid(X.dot(theta))
	J = (1. / m) * np.sum((-y * np.log(h) - (1. - y) * np.log(1. - h)))
	return J

def compute_grad(theta, X, y):
	if len(theta.shape) == 1:
		theta = theta.reshape(theta.shape[0], 1)
	m = len(y) # number of training examples
	h = sigmoid(X.dot(theta))
	grad = 1. / m * (X.T).dot(h - y)
	return grad[:,0]

def plot_decision_boundary(theta, X, y, labels=None):
	if X.shape[1] <= 3:
	    # Only need 2 points to define a line, so choose two endpoints
	    plot_x = np.array((np.min(X[:,1]) - 2, np.max(X[:,1]) + 2))

	    # Calculate the decision boundary line
	    plot_y = (-1. / theta[2]) * (theta[1] * plot_x + theta[0])

	    # Plot, and adjust axes for better viewing
	    plt.plot(plot_x, plot_y, lw=2, label=labels[-1])
	    
	    # Legend, specific for the exercise
	    plt.xlim([28, 100])
	    plt.ylim([28, 120])
	    plt.legend(fontsize=12, loc=1, frameon=True, framealpha=0.7, edgecolor='k')
	    plt.show()
	else:
	    # Here is the grid range
	    u = np.linspace(-1, 1.5, 50)
	    v = np.linspace(-1, 1.5, 50)

	    z = np.zeros((len(u), len(v)))
	    theta = theta.reshape(theta.shape[0], 1)
	    # Evaluate z = theta*x over the grid
	    for i in range(0, len(u)):
	        for j in range(0, len(v)):
	        	new = np.dot(map_feature(u[i], v[j]), theta)
	        	z[i,j] = new

	    z = z.T # important to transpose z before calling contour

	    # Plot z = 0 -- Notice you need to specify the range [0, 0] (levels=[0] in python)
	    contour = plt.contour(u, v, z, levels=[0], colors='red', linewidths=2)
	    contour.collections[0].set_label(labels[-1])
	    plt.legend(fontsize=12, loc=1, frameon=True, framealpha=0.7, edgecolor='k')
	    plt.show()

def map_feature(x1, x2):
	degree = 6
	if len(x1.shape) == 0:
		out = np.ones((1,1))
	else:
		out = np.ones((x1.shape[0],1))

	for i in range(1, degree+1):
		for j in range(0, i+1):
			col = np.array((x1**(i-j)) * (x2**j))
			if len(col.shape) >= 1:
				col = col.reshape(col.shape[0], 1)
			else:
				col = col.reshape(1,1)
			out = np.append(out, col, axis=1)
	return out

def predict(theta, X):
	m = X.shape[0] # Number of training examples
	h = sigmoid(np.dot(X, theta))
	p2 = np.asarray([1. if h[i] >= 0.5 else 0. for i in range(m)])
	p = p2.reshape(p2.shape[0], 1)

	return p



def part_one_linear(X, y):
	print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')
	plt.figure(1)
	plot_data(X, y, labels=['Admitted', 'Not Admitted'])
	
	plt.legend(fontsize=12, loc=1, frameon=True, framealpha=0.7, edgecolor='k')
	plt.xlabel('Exam 1 Score', size=18)
	plt.ylabel('Exam 2 Score', size=18)
	plt.show()

	#raw_input('Program paused. Press key to continue.\n')

def part_two_linear(X, y, initial_theta, print_output=True):
	m, n = X.shape

	# Add intercept term to x and X_test
	X = np.append(np.ones([m,1]), X, axis=1)

	# Initialize fitting parameters
	initial_theta = initial_theta.reshape(initial_theta.shape[0], 1)

	# Compute and display initial cost and gradient
	y = y[:, None] # turn into column vector
	#cost, grad = cost_function(initial_theta, X, y)
	cost = compute_cost(initial_theta, X, y)
	grad = compute_grad(initial_theta, X, y)

	if print_output:
		print('Cost at initial theta (zeros): {:f}'.format(cost))
		print('Expected cost (approx): 0.693')
		print('Gradient at initial theta (zeros):')
		print(' {}'.format(grad))
		print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')


	# Compute and display cost and gradient with non-zero theta
	test_theta = np.array([[-24], [0.2], [0.2]])
	#cost, grad = cost_function(test_theta, X, y)
	cost = compute_cost(test_theta, X, y)
	grad = compute_grad(test_theta, X, y)

	if print_output:
		print('\nCost at test theta: {:f}'.format(cost))
		print('Expected cost (approx): 0.218')
		print('Gradient at test theta:')
		print(' {}'.format(grad))
		print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

	#raw_input('Program paused. Press key to continue.\n')

	return X, y

def part_three_linear(X, y, theta, print_output=True):

	#  Set options for minimization function
	kwargs = {'maxiter': 400, 'args': (X, y), 'full_output': True}

	##  This function will return theta and the cost 
	## Tried a few different minimization functions in python. They all worked
	#theta, nf, rc = optimize.fmin_tnc(func=compute_cost, x0=theta, fprime=compute_grad, args=(X, y))
	#theta, nf, rc = optimize.fmin_tnc(func=cost_function, x0=theta, args=(X, y))
	#cost = compute_cost(theta, X, y)
	#theta, cost, go, bo, nf, ng, w = optimize.fmin_bfgs(f=compute_cost, x0=theta, fprime=compute_grad, **kwargs)
	theta, cost, nf, gf, hf, w = optimize.fmin_ncg(f=compute_cost, x0=theta, fprime=compute_grad, **kwargs)

	# Print theta to screen
	if print_output:
		print('Cost at theta found by fminunc: {:f}'.format(cost))
		print('Expected cost (approx): 0.203')
		print('theta:')
		print(' {}'.format(theta))
		print('Expected theta (approx):\n -25.161\n 0.206\n 0.201\n')

	# Plot Boundary
	plot_decision_boundary(theta, X, y, labels=['Admitted', 'Not Admitted', 'Decision Boundary'])

	# Put some labels 
	plt.xlabel('Exam 1 Score', size=18)
	plt.ylabel('Exam 2 Score', size=18)
	plt.gca().tick_params(labelsize=14)
	plt.show()

	#raw_input('Program paused. Press key to continue.\n')
	return theta

def part_four_linear(X, y, theta, print_output=True):
	prob = sigmoid(np.array([1, 45, 85]).dot(theta))
	if print_output:
		print('For a student with scores 45 and 85, we predict an admission probability of {:f}\n'.format(prob))
		print('Expected value: 0.775 +/- 0.002\n\n')

	# Compute accuracy on our training set
	p = predict(theta, X)
	if print_output:
		print('Train Accuracy: {:f}'.format(np.mean((p == y)) * 100))
		print('Expected accuracy (approx): 89.0\n')
		print('\n')

def linear():
	## Load Data
	#  The first two columns contains the exam scores and the third column
	#  contains the label.
	print('Linear!')
	data = np.loadtxt('ex2data1.txt', delimiter=',')
	X = data[:, :2]
	y = data[:, 2,]

	initial_theta = np.zeros(X.shape[1]+1)


	## ==================== Part 1: Plotting ====================
	#  We start the exercise by first plotting the data to understand the 
	#  the problem we are working with.
	part_one_linear(X, y)

	## ============ Part 2: Compute Cost and Gradient ============
	#  In this part of the exercise, you will implement the cost and gradient
	#  for logistic regression. You neeed to complete the code in 
	#  costFunction.m

	#  Setup the data matrix appropriately, and add ones for the intercept term
	X, y = part_two_linear(X, y, initial_theta, print_output=True)
	

	## ============= Part 3: Optimizing using fminunc  =============
	#  In this exercise, you will use a built-in function (fminunc) to find the
	#  optimal parameters theta.

	theta = part_three_linear(X, y, initial_theta, print_output=True)


	## ============== Part 4: Predict and Accuracies ==============
	#  After learning the parameters, you'll like to use it to predict the outcomes
	#  on unseen data. In this part, you will use the logistic regression model
	#  to predict the probability that a student with score 45 on exam 1 and 
	#  score 85 on exam 2 will be admitted.
	#
	#  Furthermore, you will compute the training and test set accuracies of 
	#  our model.
	#
	#  Your task is to complete the code in predict.m

	#  Predict probability for a student with score 45 on exam 1 
	#  and score 85 on exam 2 
	part_four_linear(X, y, theta)


def cost_function_reg(theta, X, y, llambda):
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

def compute_cost_reg(theta, X, y, llambda):
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

	return J

def compute_grad_reg(theta, X, y, llambda):
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


def part_one_regular(X, y, print_output=True):

	X = map_feature(X[:,0], X[:, 1])

	# Initialize fitting parameters
	initial_theta = np.zeros((X.shape[1], 1))

	# Set regularization parameter lambda to 1
	llambda = 1

	# Compute and display initial cost and gradient for regularized logistic
	# regression
	cost, grad = cost_function_reg(initial_theta, X, y, llambda)

	if print_output:
		print('Cost at initial theta (zeros): {:}'.format(cost))
		print('Expected cost (approx): 0.693')
		print('Gradient at initial theta (zeros) - first five values only:')
		print(' {}'.format(grad[0:5]))
		print('Expected gradients (approx) - first five values only:\n')
		print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')

		#raw_input('Program paused. Press key to continue.\n')


	# Compute and display cost and gradient
	# with all-ones theta and lambda = 10
	test_theta = np.ones((X.shape[1], 1))
	cost, grad = cost_function_reg(test_theta, X, y, 10)

	if print_output:
		print('Cost at test theta (with lambda = 10): {:}'.format(cost))
		print('Expected cost (approx): 3.16')
		print('Gradient at test theta - first five values only:')
		print(' {} '.format(grad[0:5]))
		print('Expected gradients (approx) - first five values only:\n')
		print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n')

		#raw_input('Program paused. Press key to continue.\n')

	return X, y

def part_two_regular(X, y, print_output=True):
	# Initialize fitting parameters
	initial_theta = np.zeros((X.shape[1], 1))

	# Set regularization parameter lambda to 1 (you should vary this)
	llambda = 1

	# Optimize
	kwargs = {'maxiter': 400, 'args': (X, y, llambda), 'full_output': True}
	theta, cost, nf, gf, hf, w = optimize.fmin_ncg(f=compute_cost_reg, x0=initial_theta, fprime=compute_grad_reg, **kwargs)

	# Plot Boundary
	plot_decision_boundary(theta, X, y, labels=['y=1', 'y=0', 'Decision Boundary'])

	# Labels and Legend
	plt.title('lambda = {}'.format(llambda))
	plt.xlabel('Microchip Test 1', size=18)
	plt.ylabel('Microchip Test 2', size=18)
	plt.gca().tick_params(labelsize=14)
	plt.show()

	# Compute accuracy on our training set
	p = predict(theta, X)
	if len(y.shape) == 1:
		y = y.reshape(y.shape[0], 1)
	if print_output:
		print('Train Accuracy: {:f}'.format(np.mean((p == y)) * 100))
		print('Expected accuracy (with lambda = 1): 83.1 (approx)\n')
	
def regular():
	""" Machine Learning Online Class - Exercise 2: Logistic Regression
	
	  Instructions
	  ------------
	
	  This file contains code that helps you get started on the second part
	  of the exercise which covers regularization with logistic regression.
	
	  You will need to complete the following functions in this exericse:
	
	     sigmoid.m
	     costFunction.m
	     predict.m
	     costFunctionReg.m
	
	  For this exercise, you will not need to change any code in this file,
	  or any other files other than those mentioned above.
	"""

	# Initialization
	
	print('Regular!')
	## Load Data
	#  The first two columns contains the X values and the third column
	#  contains the label (y).
	data = np.loadtxt('ex2data2.txt', delimiter=',')
	X = data[:, :2]
	y = data[:, 2,]
	
	plt.figure(2)
	plot_data(X, y, labels=['y=1', 'y=0'])

	plt.xlabel('Microchip Test 1', size=18)
	plt.ylabel('Microchip Test 2', size=18)
	plt.show()

	#raw_input('Program paused. Press key to continue.\n')


	## =========== Part 1: Regularized Logistic Regression ============
	#  In this part, you are given a dataset with data points that are not
	#  linearly separable. However, you would still like to use logistic
	#  regression to classify the data points.
	#
	#  To do so, you introduce more features to use -- in particular, you add
	#  polynomial features to our data matrix (similar to polynomial
	#  regression).
	#

	# Add Polynomial Features

	# Note that mapFeature also adds a column of ones for us, so the intercept
	# term is handled
	X, y = part_one_regular(X, y)

	## ============= Part 2: Regularization and Accuracies =============
	#  Optional Exercise:
	#  In this part, you will get to try different values of lambda and
	#  see how regularization affects the decision coundart
	#
	#  Try the following values of lambda (0, 1, 10, 100).
	#
	#  How does the decision boundary change when you vary lambda? How does
	#  the training set accuracy vary?
	#
	part_two_regular(X, y)
	



if __name__ == '__main__':
	linear()
	regular()