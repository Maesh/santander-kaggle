"""
Functions for the Santander competition.

by Ryan Gooch, Mar 2016
"""

def spliteven(X, y, bootstrap = False, size = 1.00) :
	"""
	Evenly splits data so that each class is of
	equal frequency. Expectation is that y are 
	labels and X is mice data. y should have values
	0 and 1. For now, 1 is expected to be minority
	class.

	bootstrap is boolean and if True, samples arrays
	with replacement. size is a value in range (0,1]
	that determines percentage of smallest behavior
	count to be kept. ie, if there are 30000 minority
	class segments, then size = 0.67 would mean 20000
	segments for each class are returned
	"""
	# Number of rem, wake, nrem segments
	zerocount = y[y==0].shape[0]
	onecount = y[y==1].shape[0]

	# number of segments in each class to keep
	segmentcount = int(onecount * size)

	# perform sampling by determining indices
	if bootstrap == True :
		zeroinx = np.random.choice(zerocount,
			size = segmentcount, replace = True)
		oneinx = np.random.choice(onecount,
			size = segmentcount, replace = True)
	elif bootstrap == False :
		zeroinx = np.random.choice(zerocount,
			size = segmentcount, replace = False)
		oneinx = np.random.choice(onecount,
			size = segmentcount, replace = False)
	else :
		raise ValueError('bootstrap argument must be either True or False')

	# New array for returned values
	newX = np.empty((segmentcount*3,X.shape[1]))
	newy = np.empty((segmentcount*3,))
	# drop in kept behavior segments
	newX[segmentcount:2*segmentcount,:]	= X[y[y==0]][zeroinx]
	newX[2*segmentcount:,:] 			= X[y[y==1]][oneinx]

	newy[segmentcount:2*segmentcount]	= y[y==0][zeroinx]
	newy[2*segmentcount:] 				= y[y==1][oneinx]
	# return the new matrices
	return newX.astype(int), newy.astype(int) # make sure they're int