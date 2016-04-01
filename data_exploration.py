"""
Exploration of the features and data.

Ryan Gooch, Apr 2016
"""

import numpy as np
import pandas as pd
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

from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
y = df_train['TARGET']
X = df_train.drop(['ID','TARGET'], axis=1)
print(X.shape)

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
print(X_new.shape)

"""
L1-SVM dropped features from 307 to 65. Pretty steep
"""

# Standardize, ignore numerical warning for now
X = preprocessing.scale(X_new)

# Random state for repeatability, split into training and validation sets
rs = 19683
X_train, X_val, y_train, y_val = \
		cross_validation.train_test_split(X, y, \
			test_size=0.25, random_state=rs)

import xgboost as xgb

### load data in do training
dtrain = xgb.DMatrix(X_train,label=y_train)
dtest = xgb.DMatrix(X_val,label=y_val)
param = {'max_depth':4, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
watchlist  = [(dtest,'eval'), (dtrain,'train')]
num_round = 3
bst = xgb.train(param, dtrain, num_round, watchlist)

print ('start testing prediction from first n trees')
### predict using first 1 tree
label = y_val
ypred1 = bst.predict(dtest, ntree_limit=1)
# by default, we predict using all the trees
ypred2 = bst.predict(dtest)
print ('error of ypred1=%f' % (np.sum((ypred1>0.5)!=label) /float(len(label))))
print ('error of ypred2=%f' % (np.sum((ypred2>0.5)!=label) /float(len(label))))

roc = metrics.roc_auc_score(y_val, ypred1)
print("ROC, ypred1: ", roc)
roc = metrics.roc_auc_score(y_val, ypred2)
print("ROC, ypred2: ", roc)

"""
('ROC, ypred1: ', 0.78292686247235677)
('ROC, ypred2: ', 0.80085763992638526)

So, this is good. Highly correlated, so just use ypred2 if in ensemble
"""
