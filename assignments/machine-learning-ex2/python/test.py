import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import scipy.optimize as opt
from pdb import set_trace

def sigmoid(z):  
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))


def gradient(theta, X, y):  
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)

    return grad


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




path = 'ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])  
data.head()

data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols]

# convert to numpy arrays and initalize the parameter array theta
X = np.array(X.values)  
y = np.array(y.values)  
theta = np.zeros(3)  


J1 = cost(theta, X, y)
grad1 = gradient(theta, X, y)

J2 = compute_cost(theta, X, y)
grad2 = compute_grad(theta, X, y)



set_trace()
result = opt.fmin_tnc(func=compute_cost, x0=theta, fprime=compute_grad, args=(X, y))  
J = cost(result[0], X, y)
print(J)