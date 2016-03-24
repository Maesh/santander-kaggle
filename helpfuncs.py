"""
Functions for the Santander competition.

by Ryan Gooch, Mar 2016
"""

def spliteven(X, y, bootstrap = False, size = 0.67) :
	"""
	Evenly splits data so that each class is of
	equal frequency. Expectation is that y are 
	labels and X is mice data. for y: 1 is wake,
	2 is nrem, 3 is rem, so the limiting factor 
	is rem. This would be another candidate for
	refactoring, but for now it seems fine.

	bootstrap is boolean and if True, samples arrays
	with replacement. size is a value in range (0,1]
	that determines percentage of smallest behavior
	count to be kept. ie, if there are 30000 rem 
	segments, then size = 0.67 would mean 20000 behavior
	segments for each behavior are returned
	"""
	# Number of rem, wake, nrem segments
	remcount = y[y==3].shape[0]
	nremcount = y[y==2].shape[0]
	wakecount = y[y==1].shape[0]

	# number of segments in each class to keep
	segmentcount = int(remcount * size)

	# perform sampling by determining indices
	if bootstrap == True :
		reminx = np.random.choice(remcount,
			size = segmentcount, replace = True)
		nreminx = np.random.choice(nremcount,
			size = segmentcount, replace = True)
		wakeinx = np.random.choice(wakecount,
			size = segmentcount, replace = True)
	elif bootstrap == False :
		reminx = np.random.choice(remcount,
			size = segmentcount, replace = False)
		nreminx = np.random.choice(nremcount,
			size = segmentcount, replace = False)
		wakeinx = np.random.choice(wakecount,
			size = segmentcount, replace = False)
	else :
		raise ValueError('bootstrap argument must be either True or False')

	# New array for returned values
	newX = np.empty((segmentcount*3,X.shape[1]))
	newy = np.empty((segmentcount*3,))
	# drop in kept behavior segments
	newX[:segmentcount,:] 				= X[y[y==3]][reminx]
	newX[segmentcount:2*segmentcount,:]	= X[y[y==2]][nreminx]
	newX[2*segmentcount:,:] 			= X[y[y==1]][wakeinx]

	newy[:segmentcount] 				= y[y==3][reminx]
	newy[segmentcount:2*segmentcount]	= y[y==2][nreminx]
	newy[2*segmentcount:] 				= y[y==1][wakeinx]
	# return the new matrices
	return newX.astype(int), newy.astype(int) # make sure they're int