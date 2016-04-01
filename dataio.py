# Data IO script based on work of Yuan Sun, who posted his script in the Kaggle
# forum
#
# Edited by Ryan Gooch, Mar 2016

from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.svm import OneClassSVM

def getdata(trainpath = 'train.csv',testpath = 'test.csv') :
	"""
	Imports data in pandas dataframe format. Takes path to 
	data. May need to alter this depending on system of use.
	"""
	# load data
	df_train = pd.read_csv(trainpath)
	df_test = pd.read_csv(testpath)

	# remove constant columns
	remove = []
	for col in df_train.columns:
	    if df_train[col].std() == 0:
	        remove.append(col)

	df_train.drop(remove, axis=1, inplace=True)
	df_test.drop(remove, axis=1, inplace=True)

	# remove duplicated columns
	remove = []
	c = df_train.columns
	for i in range(len(c)-1):
	    v = df_train[c[i]].values
	    for j in range(i+1,len(c)):
	        if np.array_equal(v,df_train[c[j]].values):
	            remove.append(c[j])

	df_train.drop(remove, axis=1, inplace=True)
	df_test.drop(remove, axis=1, inplace=True)

	return df_train, df_test

def writesub(id_test, y_pred, sub = "submission.csv") :
	"""
	Writes submission to Kaggle.
	"""
	submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred})
	submission.to_csv(sub, index=False)