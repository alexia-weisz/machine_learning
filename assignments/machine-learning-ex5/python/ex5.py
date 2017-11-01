import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from pdb import set_trace

## Machine Learning Online Class
#  Exercise 5 | Regularized Linear Regression and Bias-Variance
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     linearRegCostFunction.m
#     learningCurve.m
#     validationCurve.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Initialization
plt.close('all')

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  The following code will load the dataset into your environment and plot
#  the data.
#

# Load Training Data
print('Loading and Visualizing Data ...')

# Load from ex5data1: 
# You will have X, y, Xval, yval, Xtest, ytest in your environment
data = loadmat('ex5data1.mat')
X, y = data['X'], data['y']


# m = Number of examples
m = X.shape[0]

# Plot training data
plt.plot(X, y, color='r', marker='o', ms=10, lw=0)
plt.xlabel('Change in water level (x)', size=16)
plt.ylabel('Water flowing out of the dam (y)', size=16)
plt.show()

input('Program paused. Press enter to continue.\n')

## =========== Part 2: Regularized Linear Regression Cost =============
#  You should now implement the cost function for regularized linear 
#  regression. 
#

theta = np.ones((2,1))
#-->J = linearRegCostFunction([ones(m, 1) X], y, theta, 1);

print('Cost at theta = [1 ; 1] {:f}\n(this value should be about 303.993192)\n'.format(J))

raw_input('Program paused. Press enter to continue.\n')

## =========== Part 3: Regularized Linear Regression Gradient =============
#  You should now implement the gradient for regularized linear 
#  regression.
#

theta = np.ones((2,1))
#-->[J, grad] = linearRegCostFunction([ones(m, 1) X], y, theta, 1);

print('Gradient at theta = [1;1]: [{:f};{:f}]\n(this value should be about [-15.303016; 598.250744])\n'.format(grad[0], grad[1]))

raw_input('Program paused. Press enter to continue.\n')


## =========== Part 4: Train Linear Regression =============
#  Once you have implemented the cost and gradient correctly, the
#  trainLinearReg function will use your cost function to train 
#  regularized linear regression.
# 
#  Write Up Note: The data is non-linear, so this will not give a great 
#                 fit.
#

#  Train linear regression with lambda = 0
llambda = 0
#-->[theta] = trainLinearReg([np.ones((m, 1)) X], y, llambda);

#  Plot fit over the data
plt.plot(X, y, 'rx', ms=10, lw=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')

plt.plot(X, [np.ones((m, 1)), X]*theta, '--', lw=2)


raw_input('Program paused. Press enter to continue.\n')


## =========== Part 5: Learning Curve for Linear Regression =============
#  Next, you should implement the learningCurve function. 
#
#  Write Up Note: Since the model is underfitting the data, we expect to
#                 see a graph with "high bias" -- Figure 3 in ex5.pdf 
#

llambda = 0
#-->[error_train, error_val] = learningCurve([np.ones((m, 1)), X], y, [np.ones((size(Xval, 1), 1)), Xval], yval, llambda)

#-->plt.plot(1:m, error_train, 1:m, error_val)
plt.title('Learning curve for linear regression')
plt.legend('Train', 'Cross Validation')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.xlim([0, 13])
plt.ylim([0, 150])

fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i in range(m):
    print('\t{:d}\t\t{:f}\t{:f}\n'.format(i+1, error_train[i], error_val[i]))

raw_input('Program paused. Press enter to continue.\n')

## =========== Part 6: Feature Mapping for Polynomial Regression =============
#  One solution to this is to use polynomial regression. You should now
#  complete polyFeatures to map each example into its powers
#

p = 8

# Map X onto Polynomial Features and Normalize
#-->X_poly = polyFeatures(X, p);
#-->[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = np.append(np.ones((m, 1)), X_poly, axis=1) # Add Ones

# Map X_poly_test and normalize (using mu and sigma)
#-->X_poly_test = polyFeatures(Xtest, p);
#-->X_poly_test = bsxfun(@minus, X_poly_test, mu);
#-->X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);  # Add Ones
X_poly_test = np.append(np.ones((X_poly_test.shape[0], 1)), X_poly_test, axis=1) # Add ones

# Map X_poly_val and normalize (using mu and sigma)
#-->X_poly_val = polyFeatures(Xval, p);
#-->X_poly_val = bsxfun(@minus, X_poly_val, mu);
#-->X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = np.append(np.ones((X_poly_val.shape[0], 1)), X_poly_val, axis=1)# Add Ones

print('Normalized Training Example 1:')
print('  {:f}'.format(X_poly[0,:]))

raw_input('Program paused. Press enter to continue.\n')


## =========== Part 7: Learning Curve for Polynomial Regression =============
#  Now, you will get to experiment with polynomial regression with multiple
#  values of lambda. The code below runs polynomial regression with 
#  lambda = 0. You should try running the code with different values of
#  lambda to see how the fit and learning curve change.
#

llambda = 0
#-->[theta] = trainLinearReg(X_poly, y, lambda);

# Plot training data and fit
plt.figure(1)
plt.plot(X, y, 'rx', ms=10, lw=1.5)
#-->plotFit(min(X), max(X), mu, sigma, theta, p)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title('Polynomial Regression Fit (lambda = {:f})'.format(llambda))

plt.figure(2)
#-->[error_train, error_val] = learningCurve(X_poly, y, X_poly_val, yval, llambda)
#-->plt.plot(1:m, error_train, 1:m, error_val)

plt.title('Polynomial Regression Learning Curve (lambda = {:f})'.format(llambda))
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.xlim([0,13])
plt.ylim([0, 100])
plt.legend('Train', 'Cross Validation')

print('Polynomial Regression (lambda = {:f})'.format(llambda))
print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(m):
    print('  \t{:d}\t{:f}\t{:f}\n'.format(i+1, error_train[i], error_val[i]))

raw_input('Program paused. Press enter to continue.\n')

## =========== Part 8: Validation for Selecting Lambda =============
#  You will now implement validationCurve to test various values of 
#  lambda on a validation set. You will then use this to select the
#  "best" lambda value.
#

#-->[lambda_vec, error_train, error_val] = validationCurve(X_poly, y, X_poly_val, yval)

plt.close('all')
plt.plot(lambda_vec, error_train, lambda_vec, error_val)
plt.legend('Train', 'Cross Validation')
plt.xlabel('lambda')
plt.ylabel('Error')

print('lambda\t\tTrain Error\tValidation Error')
for i in range(len(lambda_vec)):
	print(' {:f}\t{:f}\t{:f}\n'.format(lambda_vec[i], error_train[i], error_val[i]))

raw_input('Program paused. Press enter to continue.\n')
