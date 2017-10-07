import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace

def plot_data(x, y):
	admitted = (y == 1)
	not_admitted = (y == 0)

	plt.plot(x[admitted, 0], x[admitted, 1], 'kP', label='Admitted')
	plt.plot(x[not_admitted, 0], x[not_admitted, 1], 'yo', mec='k', mew=0.5, label='Not Admitted')
	plt.legend(fontsize=12, loc=1, frameon=True, framealpha=0.7, edgecolor='k')

	plt.xlabel('Exam 1 Score', size=18)
	plt.ylabel('Exam 2 Score', size=18)


	plt.show()

def sigmoid(z):
	return 1. / (1 +  np.exp(-z))


def cost_function(theta, X, y):
	# Initialize some useful values
	m = len(y) # number of training examples

	# ====================== YOUR CODE HERE ======================
	# Instructions: Compute the cost of a particular choice of theta.
	#               You should set J to the cost.
	#               Compute the partial derivatives and set grad to the partial
	#               derivatives of the cost w.r.t. each parameter in theta
	#
	# Note: grad should have the same dimensions as theta
	#
	h = sigmoid(X.dot(theta))
	J = (1. / m) * np.sum((-y * np.log(h) - (1. - y) * np.log(1. - h)))
	grad = (1. / m) * np.sum(((h-y) * X))
	return J, grad


def main():
	## Load Data
	#  The first two columns contains the exam scores and the third column
	#  contains the label.

	data = np.loadtxt('ex2data1.txt', delimiter=',')
	X = data[:, :2]
	y = data[:, 2,]


	## ==================== Part 1: Plotting ====================
	#  We start the exercise by first plotting the data to understand the 
	#  the problem we are working with.

	print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n');

	plot_data(X, y)

	raw_input('Press key to continue...\n')


	## ============ Part 2: Compute Cost and Gradient ============
	#  In this part of the exercise, you will implement the cost and gradient
	#  for logistic regression. You neeed to complete the code in 
	#  costFunction.m

	#  Setup the data matrix appropriately, and add ones for the intercept term
	m, n = X.shape

	# Add intercept term to x and X_test
	X = np.append(np.ones([m,1]), X, axis=1)

	# Initialize fitting parameters
	initial_theta = np.zeros([n+1, 1])

	# Compute and display initial cost and gradient
	y = y[:, None] # turn into column vector
	cost, grad = cost_function(initial_theta, X, y)

	print('Cost at initial theta (zeros): {:f}'.format(cost))
	print('Expected cost (approx): 0.693')
	print('Gradient at initial theta (zeros):')
	print(' {:f}'.format(grad))
	print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')
	raw_input('waiting...\n')
	# Compute and display cost and gradient with non-zero theta
	test_theta = np.array([[-24], [0.2], [0.2]])
	cost, grad = cost_function(test_theta, X, y)


	print('\nCost at test theta: {:f}\n'.format(cost))
	print('Expected cost (approx): 0.218\n')
	print('Gradient at test theta: \n')
	print(' {:f}\n'.format(grad))
	print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

	print('\nProgram paused. Press enter to continue.\n')
	raw_input('Press key to continue...\n')


if __name__ == '__main__':
	main()
