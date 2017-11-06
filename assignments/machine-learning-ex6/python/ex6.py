import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from pdb import set_trace



def plotData(X, y):
	pos = np.where(y == 1)
	neg = np.where(y == 0)

	# Plot Examples
	plt.plot(X[pos, 0], X[pos, 1], 'ko',lw=1, ms=7, mew=0.3, mec='0.5')
	plt.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', mec='0.5', mew=0.3, ms=7)
	plt.show()

def visualizeBoundaryLinear(X, y, model):
	w = model.coef_[0]
	b = model.intercept_[0]

	xp = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
	yp = -(w[0] * xp + b) / w[1]
	plotData(X, y)
	plt.plot(xp, yp, '-b')
	plt.draw()

def gaussianKernel(x1, x2, sigma):
	sim = np.exp(-np.sum((x1 - x2)**2) / (2 * sigma**2))
	return sim

def visualizeBoundary(X, y, model, color='b', lw=2):
	plotData(X, y)

	# Make classification predictions over a grid of values
	x1plot = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)[:, np.newaxis]
	x2plot = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 100)[:, np.newaxis]
	X1, X2 = np.meshgrid(x1plot, x2plot)
	vals = np.zeros(X1.shape)
	for i in range(X1.shape[1]):
		this_X = np.append(X1[:, i][:, np.newaxis], X2[:, i][:, np.newaxis], axis=1)
		vals[:, i] = model.predict(this_X)

	# Plot the SVM boundary
	cs = plt.contour(X1, X2, vals, levels=[0.5], colors=(color), linewidths=(lw))
	plt.draw()


def dataset3Params(X, y, Xval, yval):
	vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
	m = len(vals)
	pred_err = np.zeros((m*m, 4))
	for i in range(m):
		for j in range(m):
			k = i*m+j
			C = vals[i]
			sigma = vals[j]
			gamma = 1. / ( 2 * sigma**2)
			model = svm.SVC(C=C, kernel='rbf', gamma=gamma)
			model.fit(X, y[:,0])

			predictions = model.predict(Xval)
			err = np.mean((predictions != yval[:,0]))
			pred_err[k,:] = [C, sigma, gamma, err]
	
	min_err = np.min(pred_err[:,3])
	min_row = np.where(pred_err[:,3] == min_err)[0]
	best_C = pred_err[min_row, 0]
	best_sigma = pred_err[min_row, 1]
	best_gamma = pred_err[min_row, 2]

	return best_C[0], best_sigma[0], best_gamma[0]




#############
plt.close()

## PART 1 ##
print('Loading and Visualizing Data ...')

# Load from ex6data1: 
# You will have X, y in your environment
data = loadmat('ex6data1.mat')
X, y = data['X'], data['y']

# Plot training data
plotData(X, y);

#input('Program paused. Press enter to continue.')



## PART 2 ##
print('Training Linear SVM ...')

# You should try to change the C value below and see how the decision
# boundary varies (e.g., try C = 1000)
C = 1
model = svm.SVC(C=C, kernel='linear', tol=1e-3, max_iter=20)
model.fit(X, y)
#model = svm.SVC(C=C, X, y, C, @linearKernel, 1e-3, 20)

visualizeBoundaryLinear(X, y, model)

#input('Program paused. Press enter to continue.')



## PART 3 ###
print('\nEvaluating the Gaussian Kernel ...')

x1 = np.asarray([1, 2, 1])
x2 = np.asarray([0, 4 ,-1])
sigma = 2
sim = gaussianKernel(x1, x2, sigma)

print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = {:f} :'.format(sigma))
print('\t{:f}\n(for sigma = 2, this value should be about 0.324652)'.format(sim))

#input('Program paused. Press enter to continue.')
plt.close()



## PART 4 ##
print('Loading and Visualizing Data ...')

# Load from ex6data2: 
# You will have X, y in your environment
data = loadmat('ex6data2.mat')
X, y = data['X'], data['y']

# Plot training data
plotData(X, y)

#input('Program paused. Press enter to continue.')



## PART 5 ##
print('Training SVM with RBF Kernel')

# SVM Parameters
C = 1
sigma = 0.1
gamma = 1. / (2 * sigma**2)

# We set the tolerance and max_passes lower here so that the code will run
# faster. However, in practice, you will want to run the training to
# convergence.
model = svm.SVC(C=C, kernel='rbf', gamma=gamma)
model.fit(X, y)

visualizeBoundary(X, y, model)

#input('Program paused. Press enter to continue.\n')
plt.close()


## PART 6 ##
print('Loading and Visualizing Data ...')

# Load from ex6data3: 
# You will have X, y in your environment
data = loadmat('ex6data3.mat')
X, y = data['X'], data['y']

# Plot training data
plotData(X, y)

#input('Program paused. Press enter to continue.')



## PART 7 ##
Xval, yval = data['Xval'], data['yval']

# Try different SVM Parameters here
vals = np.asarray([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])

parameters={'C': vals, 'gamma': 1./(2. * vals**2)}
svc = svm.SVC(kernel='rbf')
model = GridSearchCV(svc, parameters)

# Train the SVM

model.fit(Xval, yval[:,0])

C_best = model.best_params_['C']
gamma_best = model.best_params_['gamma']
sigma_best = np.sqrt(1./(2*gamma_best))




C, sigma, gamma = dataset3Params(X, y, Xval, yval)
new_model = svm.SVC(C=C, kernel='rbf', gamma=gamma)
new_model.fit(X, y)
visualizeBoundary(X, y, new_model)





