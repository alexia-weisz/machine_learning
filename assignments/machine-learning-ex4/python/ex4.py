import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import random
from pdb import set_trace
"""
 Machine Learning Online Class - Exercise 4 Neural Network Learning

  Instructions
  ------------
 
  This file contains code that helps you get started on the
  linear exercise. You will need to complete the following functions 
  in this exericse:

     sigmoidGradient.m
     randInitializeWeights.m
     nnCostFunction.m

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


def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, llambda):
  pass

def rand_initialize_weights(input_layer_size, hidden_layer_size):
  pass

def check_nn_gradients():
  pass

# Initialization
plt.close('all')

# Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   
                         # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.


# Load Training Data
print('Loading and Visualizing Data ...')

data = loadmat('ex4data1.mat')
X = data['X']
m = X.shape[0]

# Randomly select 100 data points to display
rand_indices = random.sample(range(1, m+1), m)
sel = X[rand_indices[0:100], :]

h, display_array = display_data(sel)

raw_input('Program paused. Press enter to continue.')


## ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print('Loading Saved Neural Network Parameters ...')

# Load the weights into variables Theta1 and Theta2
data = loadmat('ex4weights.mat');
Theta1, Theta2 = data['Theta1'], data['Theta2']


# Unroll parameters 
nn_params = np.append(Theta1, Theta2)
set_trace()
## ================ Part 3: Compute Cost (Feedforward) ================
#  To the neural network, you should first start by implementing the
#  feedforward part of the neural network that returns the cost only. You
#  should complete the code in nnCostFunction.m to return cost. After
#  implementing the feedforward to compute the cost, you can verify that
#  your implementation is correct by verifying that you get the same cost
#  as us for the fixed debugging parameters.
#
#  We suggest implementing the feedforward cost *without* regularization
#  first so that it will be easier for you to debug. Later, in part 4, you
#  will get to implement the regularized cost.
#
print('Feedforward Using Neural Network ...')

# Weight regularization parameter (we set this to 0 here).
llambda = 0

J = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, llambda)

print('Cost at parameters (loaded from ex4weights): {} \n(this value should be about 0.287629)'.format(J))

raw_input('Program paused. Press enter to continue.')

## =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.


print('Checking Cost Function (w/ Regularization) ...')

# Weight regularization parameter (we set this to 1 here).
llambda = 1

J = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, llambda)

print('Cost at parameters (loaded from ex4weights): {:f}\n(this value should be about 0.383770)'.format(J))

raw_input('Program paused. Press enter to continue.')


## ================ Part 5: Sigmoid Gradient  ================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.m file.


print('Evaluating sigmoid gradient...')

g = sigmoid_gradient(np.asarray([-1, -0.5, 0, 0.5, 1]))
print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n')
print('{:f} '.format(g))
print('\n\n')

raw_input('Program paused. Press enter to continue.')


## ================ Part 6: Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two
#  layer neural network that classifies digits. You will start by
#  implementing a function to initialize the weights of the neural network
#  (randInitializeWeights.m)

print('Initializing Neural Network Parameters ...')

initial_Theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
initial_Theta2 = rand_initialize_weights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.append(initial_Theta1, initial_Theta2, axis=1)


## =============== Part 7: Implement Backpropagation ===============
#  Once your cost matches up with ours, you should proceed to implement the
#  backpropagation algorithm for the neural network. You should add to the
#  code you've written in nnCostFunction.m to return the partial
#  derivatives of the parameters.
#
print('Checking Backpropagation...')

# Check gradients by running checkNNGradients
check_nn_gradients()

raw_input('Program paused. Press enter to continue.')


## =============== Part 8: Implement Regularization ===============
#  Once your backpropagation implementation is correct, you should now
#  continue to implement the regularization with the cost and gradient.
#

print('Checking Backpropagation (w/ Regularization) ...')

#  Check gradients by running checkNNGradients
llambda = 3
check_nn_gradients(llambda)

# Also output the costFunction debugging values
debug_J  = nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, llambda)

print('Cost at (fixed) debugging parameters (w/ lambda = {:f}: {:f}\n(for lambda = 3, this value should be about 0.576051)'.format(llambda, debug_J))

raw_input('Program paused. Press enter to continue.')


## =================== Part 8: Training NN ===================
#  You have now implemented all the code necessary to train a neural 
#  network. To train your neural network, we will now use "fmincg", which
#  is a function which works similarly to "fminunc". Recall that these
#  advanced optimizers are able to train our cost functions efficiently as
#  long as we provide them with the gradient computations.

print('Training Neural Network...')

#  After you have completed the assignment, change the MaxIter to a larger
#  value to see how more training helps.

#  You should also try different values of lambda
llambda = 1

# Create "short hand" for the cost function to be minimized
#%#costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, llambda)


def cost_function(p):

  def wrapper():
    return nn_cost_function()

  return wrapper




def cost_function(p):
    return nn_cost_function(p, input_layer_size, hidden_layer_size, num_labels, X, y, llambda)

# Now, costFunction is a function that takes in only one argument (the
# neural network parameters)
#[nn_params, cost] = fmincg(costFunction, initial_nn_params, options)

#kwargs = {'maxiter': 50, 'args': (X, (y==(c+1)), llambda)}
kwargs = {'maxiter': 50}
nn_params, cost = optimize.fmin_bfgs(f=cost_function, x0=initial_nn_params, **kwargs)


# Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params[1:hidden_layer_size * (input_layer_size + 1)], hidden_layer_size, (input_layer_size + 1))

Theta2 = reshape(nn_params[(1 + (hidden_layer_size * (input_layer_size + 1))):end], num_labels, (hidden_layer_size + 1))

raw_input('Program paused. Press enter to continue.')


## ================= Part 9: Visualize Weights =================
#  You can now "visualize" what the neural network is learning by 
#  displaying the hidden units to see what features they are capturing in 
#  the data.

print('Visualizing Neural Network...')

h, display_array = display_data(Theta1[:, 2:])

raw_input('Program paused. Press enter to continue.')

## ================= Part 10: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X)

fprintf('Training Set Accuracy: {:f}', np.mean((pred == y)) * 100)


