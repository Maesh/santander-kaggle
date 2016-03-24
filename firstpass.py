"""
First pass at the Santander competition.

Note: This NN doesn't do better than the all zeroes benchmark as it 
rarely predicts 1. Only 6 on the validation set and none on test set

By Ryan Gooch
March, 2016
"""

import numpy as np
import pandas as pd
import keras
import theano

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD

from sklearn import cross_validation, preprocessing, metrics

# also import data import and export functions
from dataio import getdata, writesub

# Get the data in, skip header row
# train = np.genfromtxt('train.csv',delimiter=',',skip_header=1)
df_train, df_test = getdata()

# Get target values
y = df_train['TARGET'].values

X_train = df_train.drop(['ID','TARGET'], axis=1).values

# Standardize, ignore numerical warning for now
# X = preprocessing.scale(X_train)

# Random state for repeatability, split into training and validation sets
rs = 19683
X_train, X_val, y_train, y_val = \
		cross_validation.train_test_split(X, y, \
			test_size=0.25, random_state=rs)

# one hot encode with pandas. A little messy but easy
y_train = pd.get_dummies(pd.Series(y_train)).values
y_val = pd.get_dummies(pd.Series(y_val)).values

model = Sequential()
# Trying various NN configurations, see what sticks
model.add(Dense(64, input_dim=X_train.shape[1], init='he_normal'))#, W_regularizer=l2(0.1)))
model.add(PReLU()) # Prelu works well I have found in the past
model.add(Dropout(0.5)) # Reduce overfitting
model.add(Dense(128, init='he_normal',input_dim=64))
model.add(PReLU())
model.add(Dropout(0.5))
model.add(Dense(64, init='he_normal',input_dim=128))
model.add(PReLU())
model.add(Dropout(0.5))
model.add(Dense(2, init='he_normal',input_dim=64))
model.add(Activation('softmax')) # classification softmax, regression tanh or sigmoid

# Stochastic gradient descent to train
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

# Use binary cross entropy for now for classification, adam as suggested
model.compile(loss='binary_crossentropy', optimizer='adadelta')

# Fit the model. Training on small number of epochs to start with.
f = model.fit(X_train, y_train, nb_epoch=100, shuffle=True,
	batch_size=128, validation_split=0.15,
	show_accuracy=True, verbose=1)

print("Making predictions on validation set")
# Make predictions on validation data
valid_preds = model.predict_proba(X_val, verbose=0)

# Take max value in preds rows as classification
pred = np.zeros((len(valid_preds)))
yint = np.zeros((len(valid_preds)))
for row in np.arange(0,len(valid_preds)) :
	pred[row] = np.argmax(valid_preds[row])
	yint[row] = np.argmax(y_val[row])

roc = metrics.roc_auc_score(yint, pred)
print("ROC:", roc)

print("Making predictions on test set")

id_test = df_test['ID']
X_test = df_test.drop(['ID'], axis=1).values

# make predictions using model, on test set
test_preds = model.predict_proba(X_test)

# Take max value in preds rows as classification
pred = np.zeros((len(test_preds)))
for row in np.arange(0,len(test_preds)) :
	pred[row] = np.max(test_preds[row])

# Write to file, change file name as needed
writesub(id_test, pred, sub = "NN.64.128.64.csv")