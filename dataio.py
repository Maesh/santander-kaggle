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

def getdata() :
	# load data
	df_train = pd.read_csv('train.csv')
	df_test = pd.read_csv('test.csv')

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

# y_train = df_train['TARGET'].values
# X_train = df_train.drop(['ID','TARGET'], axis=1).values

# id_test = df_test['ID']
# X_test = df_test.drop(['ID'], axis=1).values