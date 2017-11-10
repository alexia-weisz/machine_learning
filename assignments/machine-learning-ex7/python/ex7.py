import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
from sklearn import svm
from numpy.linalg import svd
from pdb import set_trace



def K_means():

	def findClosestCentroids(X, centroids):
		K = centroids.shape[0]

		# You need to return the following variables correctly.
		idx = np.zeros((X.shape[0],1))

		# ====================== YOUR CODE HERE ======================
		# Instructions: Go over every example, find its closest centroid, and store
		#               the index inside idx at the appropriate location.
		#               Concretely, idx(i) should contain the index of the centroid
		#               closest to example i. Hence, it should be a value in the 
		#               range 1..K
		#
		# Note: You can use a for-loop over the examples to compute this.
		#

		m = X.shape[0]
		for i in range(m):
			ex = X[i,:]
			new_ex = np.ones((K, 1)) * ex
			ex_diff = new_ex - centroids
			diff = np.diag(np.inner(ex_diff, ex_diff))
			iw = np.where(diff == min(diff))[0][0]
			idx[i] = iw + 1
			
		return idx.flatten()

	def computeCentroids(X, idx, K):
		# Useful variables
		m, n = X.shape

		# You need to return the following variables correctly.
		centroids = np.zeros((K, n))

		# ====================== YOUR CODE HERE ======================
		# Instructions: Go over every centroid and compute mean of all points that
		#               belong to it. Concretely, the row vector centroids(i, :)
		#               should contain the mean of the data points assigned to
		#               centroid i.
		#
		# Note: You can use a for-loop over the centroids to compute this.
		#

		for i in range(K):
			coords = X[idx==i+1,:]
			centroids[i,:] = np.mean(coords, axis=0)

		return centroids

	def runkMeans(X, initial_centroids, max_iters, plot_progress=False):
		# RUNKMEANS runs the K-Means algorithm on data matrix X, where each row of X
		# is a single example
		#   [centroids, idx] = RUNKMEANS(X, initial_centroids, max_iters, ...
		#   plot_progress) runs the K-Means algorithm on data matrix X, where each 
		#   row of X is a single example. It uses initial_centroids used as the
		#   initial centroids. max_iters specifies the total number of interactions 
		#   of K-Means to execute. plot_progress is a true/false flag that 
		#   indicates if the function should also plot its progress as the 
		#   learning happens. This is set to false by default. runkMeans returns 
		#   centroids, a Kxn matrix of the computed centroids and idx, a m x 1 
		#   vector of centroid assignments (i.e. each entry in range [1..K])
		#

		# Set default value for plot progress
		if not plot_progress:
		    plot_progress = False

		# Plot the data if we are plotting progress
		if plot_progress:
		    plt.figure()
		    plt.show()
		
		# Initialize values
		m, n = X.shape
		K = initial_centroids.shape[0]
		centroids = initial_centroids
		previous_centroids = centroids
		idx = np.zeros((m, 1))

		# Run K-Means
		for i in range(max_iters):
		    
		    # Output progress
		    print('K-Means iteration {:d}/{:d}...'.format(i+1, max_iters))
		    #if exist('OCTAVE_VERSION'):
		    #    fflush(stdout)
		    
		    # For each example in X, assign it to the closest centroid
		    idx = findClosestCentroids(X, centroids)
		    
		    # Optionally, plot progress here
		    if plot_progress:
		        plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
		        previous_centroids = centroids;
		        #input('Press enter to continue.')
		       
		    # Given the memberships, compute new centroids
		    centroids = computeCentroids(X, idx, K)

		# Re-draw if we are plotting progress
		if plot_progress:
		    plt.draw()

		return centroids, idx

	def plotProgresskMeans(X, centroids, previous, idx, K, i):
		# Plot the examples
		plotDataPoints(X, idx, K)

		# Plot the centroids as black x's
		plt.plot(centroids[:,0], centroids[:,1], 
				 marker='*', mfc='k', mec='k', ms=10, lw=0)
		plt.draw()

		# Plot the history of the centroids with lines
		for j in range(centroids.shape[0]):
		    drawLine(centroids[j, :], previous[j, :])
		plt.draw()

		# Title
		plt.title('Iteration number {:d}'.format(i+1))

	def plotDataPoints(X, idx, K):
		cmap = cm.plasma

		# Plot the data
		plt.scatter(X[:,0], X[:,1], s=20, c=idx*2, cmap=cmap, 
					edgecolor='0.5', linewidth=0.3)

	def drawLine(p1, p2):
		plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='k', lw=1)

	def kMeansInitCentroids(X, K):
		#KMEANSINITCENTROIDS This function initializes K centroids that are to be 
		#used in K-Means on the dataset X
		#   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
		#   used with the K-Means on the dataset X
		#

		# You should return this values correctly
		centroids = np.zeros((K, X.shape[1]))

		# ====================== YOUR CODE HERE ======================
		# Instructions: You should set centroids to randomly chosen examples from
		#               the dataset X
		#

		# Initialize the centroids to be random examples
		# Randomly reorder the indices of examples
		randidx = np.random.permutation(X.shape[0])
		# Take the first K examples as centroids
		centroids = X[randidx[0:K], :]

		return centroids



	plt.close('all')

	## ================= Part 1: Find Closest Centroids ====================
	#  To help you implement K-Means, we have divided the learning algorithm 
	#  into two functions -- findClosestCentroids and computeCentroids. In this
	#  part, you should complete the code in the findClosestCentroids function. 
	#
	print('Finding closest centroids.')

	# Load an example dataset that we will be using
	data = loadmat('ex7data2.mat')
	X = data['X']

	# Select an initial set of centroids
	K = 3 # 3 Centroids
	initial_centroids = np.asarray(([3, 3], [6, 2], [8, 5]))

	# Find the closest centroids for the examples using the
	# initial_centroids
	idx = findClosestCentroids(X, initial_centroids)
	print('Closest centroids for the first 3 examples:')
	print(' {:}'.format(idx[0:3]))
	print('(the closest centroids should be 1, 3, 2 respectively)')

	#input('Program paused. Press enter to continue.')


	## ===================== Part 2: Compute Means =========================
	#  After implementing the closest centroids function, you should now
	#  complete the computeCentroids function.
	#
	print('Computing centroids means.')

	# Compute means based on the closest centroids found in the previous part.
	centroids = computeCentroids(X, idx, K)

	print('Centroids computed after initial finding of closest centroids:')
	print(' {}'.format(centroids))
	print('(the centroids should be')
	print('   [ 2.428301 3.157924 ]')
	print('   [ 5.813503 2.633656 ]')
	print('   [ 7.119387 3.616684 ]')

	#input('Program paused. Press enter to continue.')


	# =================== Part 3: K-Means Clustering ======================
	#  After you have completed the two functions computeCentroids and
	#  findClosestCentroids, you have all the necessary pieces to run the
	#  kMeans algorithm. In this part, you will run the K-Means algorithm on
	#  the example dataset we have provided. 
	#
	print('Running K-Means clustering on example dataset.');

	# Load an example dataset
	data = loadmat('ex7data2.mat')
	X = data['X']

	# Settings for running K-Means
	K = 3
	max_iters = 10

	# For consistency, here we set centroids to specific values
	# but in practice you want to generate them automatically, such as by
	# settings them to be random examples (as can be seen in
	# kMeansInitCentroids).
	initial_centroids = np.asarray(([3, 3], [6, 2], [8, 5]))

	# Run K-Means algorithm. The 'true' at the end tells our function to plot
	# the progress of K-Means
	centroids, idx = runkMeans(X, initial_centroids, max_iters, plot_progress=True)
	print('K-Means Done.')

	#input('Program paused. Press enter to continue.')
	plt.close()


	## ============= Part 4: K-Means Clustering on Pixels ===============
	#  In this exercise, you will use K-Means to compress an image. To do this,
	#  you will first run K-Means on the colors of the pixels in the image and
	#  then you will map each pixel onto its closest centroid.
	#  
	#  You should now complete the code in kMeansInitCentroids.m
	#

	print('\nRunning K-Means clustering on pixels from an image.')

	#  Load an image of a bird
	A = mpimg.imread('bird_small.png')

	# If imread does not work for you, you can try instead
	#   load ('bird_small.mat');

	# this is already done in the imread step in python
	#A = A / 255. # Divide by 255 so that all values are in the range 0 - 1

	# Size of the image
	img_size = A.shape

	# Reshape the image into an Nx3 matrix where N = number of pixels.
	# Each row will contain the Red, Green and Blue pixel values
	# This gives us our dataset matrix X that we will use K-Means on.
	X = A.reshape(img_size[0] * img_size[1], 3)

	# Run your K-Means algorithm on this data
	# You should try different values of K and max_iters here
	K = 16
	max_iters = 10

	# When using K-Means, it is important the initialize the centroids
	# randomly. 
	# You should complete the code in kMeansInitCentroids.m before proceeding
	initial_centroids = kMeansInitCentroids(X, K)

	# Run K-Means
	centroids, idx = runkMeans(X, initial_centroids, max_iters)
	
	#input('Program paused. Press enter to continue.')



	## ================= Part 5: Image Compression ======================
	# In this part of the exercise, you will use the clusters of K-Means to
	#  compress an image. To do this, we first find the closest clusters for
	#  each example. After that, we 

	print('Applying K-Means to compress an image.')

	# Find closest cluster members
	idx = findClosestCentroids(X, centroids)

	# Essentially, now we have represented the image X as in terms of the
	# indices in idx. 

	# We can now recover the image from the indices (idx) by mapping each pixel
	# (specified by its index in idx) to the centroid value

	#X_recovered = centroids[idx, :]
	X_recovered = np.ones((idx.shape[0], 3))
	for i in range(idx.shape[0]):
		X_recovered[i] = centroids[int(idx[i])-1,:]

	# Reshape the recovered image into proper dimensions
	X_recovered = X_recovered.reshape(img_size[0], img_size[1], 3)

	# Display the original image 
	fig, (ax1, ax2) = plt.subplots(1, 2)
	ax1.imshow(A)
	ax1.set_title('Original')

	# Display compressed image side by side
	ax2.imshow(X_recovered)
	ax2.set_title('Compressed, with {:d} colors.'.format(K))
	plt.show()
	input('Program paused. Press enter to continue.')


def PCA():

	def featureNormalize(X):
		mu = np.mean(X, axis=0)
		X_norm = X - mu

		sigma = np.std(X_norm, axis=0);
		X_norm = X_norm / sigma

		return X_norm, mu, sigma

	def pca(X):
		m, n = X.shape

		# You need to return the following variables correctly.
		U = np.zeros(n)
		S = np.zeros(n)

		# ====================== YOUR CODE HERE ======================
		# Instructions: You should first compute the covariance matrix. Then, you
		#               should use the "svd" function to compute the eigenvectors
		#               and eigenvalues of the covariance matrix. 
		#
		# Note: When computing the covariance matrix, remember to divide by m (the
		#       number of examples).
		#

		Sigma = 1. / m * np.inner(X.T, X.T)
		U, S, V = svd(Sigma)

		return U, S

	def drawLine(p1, p2, color='k', lw=1, ls='-'):
		plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color='k', lw=1)

	def projectData(X, U, K):
		Ureduce = U[:,0:K]
		Z = np.inner(X ,Ureduce.T)
		return Z

	def recoverData(Z, U, K):
		X_rec = np.inner(Z, U[:,0:K])
		return X_rec

	plt.close('all')

	## ================== Part 1: Load Example Dataset  ===================
	#  We start this exercise by using a small dataset that is easily to
	#  visualize
	#
	print('Visualizing example dataset for PCA.')

	#  The following command loads the dataset. You should now have the 
	#  variable X in your environment
	data = loadmat('ex7data1.mat')
	X = data['X']

	#  Visualize the example dataset
	plt.plot(X[:, 0], X[:, 1], 'bo')
	plt.xlim([0.5, 6.5])
	plt.ylim([2, 8])
	plt.show()

	#input('Program paused. Press enter to continue.')



	## =============== Part 2: Principal Component Analysis ===============
	#  You should now implement PCA, a dimension reduction technique. You
	#  should complete the code in pca.m
	#
	print('Running PCA on example dataset.')

	#  Before running PCA, it is important to first normalize X
	X_norm, mu, sigma = featureNormalize(X)

	#  Run PCA
	U, S = pca(X_norm)
	S = np.diag(S)

	#  Compute mu, the mean of the each feature

	#  Draw the eigenvectors centered at mean of data. These lines show the
	#  directions of maximum variations in the dataset.
	drawLine(mu, mu + 1.5 * S[0,0] * U[:,0].T, ls='-', color='k', lw=2)
	drawLine(mu, mu + 1.5 * S[1,1] * U[:,1].T, ls='-', color='k', lw=2)

	print('Top eigenvector: ')
	print(' U(:,1) = {:.6f} {:.6f}'.format(U[0,0], U[1,0]))
	print('\n(you should expect to see -0.707107 -0.707107)')

	#input('Program paused. Press enter to continue.')


	## =================== Part 3: Dimension Reduction ===================
	#  You should now implement the projection step to map the data onto the 
	#  first k eigenvectors. The code will then plot the data in this reduced 
	#  dimensional space.  This will show you what the data looks like when 
	#  using only the corresponding eigenvectors to reconstruct it.
	#
	#  You should complete the code in projectData.m
	#
	print('Dimension reduction on example dataset.')

	#  Plot the normalized dataset (returned from pca)
	plt.plot(X_norm[:, 0], X_norm[:, 1], 'bo')
	plt.xlim([-4, 3])
	plt.ylim([-4, 3])

	#  Project the data onto K = 1 dimension
	K = 1
	Z = projectData(X_norm, U, K)

	print('Projection of the first example: {:f}'.format(Z.flatten()[0]))
	print('\n(this value should be about 1.481274)')

	X_rec  = recoverData(Z, U, K)
	print('Approximation of the first example: {} {}'.format(X_rec[0,0], X_rec[0,1]))
	print('(this value should be about  -1.047419 -1.047419)')

	#  Draw lines connecting the projected points to the original points
	plt.plot(X_rec[:, 0], X_rec[:, 1], 'ro')
	for i in range(X_norm.shape[0]):
	    drawLine(X_norm[i,:], X_rec[i,:], '--k', lw=1)


	input('Program paused. Press enter to continue.')
	



#K_means()
PCA()
