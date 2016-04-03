"""
Exploration of the features and data.

Ryan Gooch, Apr 2016
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import theano

from sklearn import cross_validation, preprocessing, metrics

# also import data import and export functions
from dataio import getdata, writesub

from sklearn.feature_selection import VarianceThreshold

# Get the data in, skip header row
# train = np.genfromtxt('train.csv',delimiter=',',skip_header=1)
trainpath = '/media/ryan/Charlemagne/kaggle/santander-kaggle/train.csv'
testpath = '/media/ryan/Charlemagne/kaggle/santander-kaggle/test.csv'
df_train, df_test = getdata(trainpath,testpath)

rs = 19683

"""
Feature Selection 
"""


from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
y = df_train['TARGET']
X = df_train.drop(['ID','TARGET'], axis=1)
print(X.shape)

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False, random_state = rs).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
print(X_new.shape)

X_IDs = df_test['ID']
X_test = df_test.drop(['ID'],axis=1)
X_test = model.transform(X_test)

"""
L1-SVM dropped features from 307 to 65. Pretty steep
"""

# Store in dataframe, loop through and categorize all remaining features
df_new = pd.DataFrame(X_new)
df_new_test = pd.DataFrame(X_test)
labels = np.arange(0,100)
for k in df_new.columns :
	if np.var(df_new_test[k]) > 0 :
		df_new_test[str(k)+'_new'] = pd.cut(df_new_test[k], \
			100, include_lowest = True,labels=labels)
		df_new_test = df_new_test.drop(k,axis=1)

		df_new[str(k)+'_new'] = pd.cut(df_new[k], \
			100, include_lowest = True,labels=labels)
		df_new = df_new.drop(k,axis=1)
	else :
		df_new_test = df_new_test.drop(k,axis=1)
		df_new = df_new.drop(k,axis=1)

# Standardize, ignore numerical warning for now
X = preprocessing.scale(df_new)
X_test = preprocessing.scale(df_new_test)

# Random state for repeatability, split into training and validation sets
rs = 19683
X_train, X_val, y_train, y_val = \
		cross_validation.train_test_split(X, y, \
			test_size=0.25, random_state=rs)
