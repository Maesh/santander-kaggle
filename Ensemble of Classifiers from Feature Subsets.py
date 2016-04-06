
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import time

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

from dataio import getdata, writesub


# In[2]:

# /media/ryan/Charlemagne/kaggle/santander-kaggle
trainpath = '/media/ryan/Charlemagne/kaggle/santander-kaggle/train.csv' 
testpath = '/media/ryan/Charlemagne/kaggle/santander-kaggle/test.csv' 
df_train,df_test = getdata(trainpath,testpath)


# In[3]:

df_features = pd.read_csv('features/feature_combos_15.csv')
df_features = df_features.drop('Unnamed: 0',axis=1)
df_features.head(5)


# In[4]:

df_features[df_features['combinedAUC']>0.835]


# In[5]:

start_time = time.time()


# In[6]:

rs = 19683

# split data into train and test
test_id = df_test.ID
test = df_test.drop(["ID"],axis=1)

X = df_train.drop(["TARGET","ID"],axis=1)
y = df_train.TARGET.values

X_train = X
y_train = y
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=rs)


# In[7]:

thresh = 0.835
df_features_greater_than = df_features[df_features['combinedAUC']>thresh]
sub = np.zeros((test.shape[0],df_features_greater_than.shape[0]))
for index, row in df_features_greater_than.iterrows():
    # grid search for params
    xgb_model = xgb.XGBClassifier()
    clf = GridSearchCV(xgb_model, {'max_depth': [2, 6, 10],
                                   'learning_rate': [0.01, 0.1, 0.25],
                                   'n_estimators': [50, 100, 200],
                                   'seed':[rs]}, 
                        verbose=0, n_jobs=-1, scoring = 'roc_auc')
    clf.fit(X_train, y_train)

    ## # Train Model
    # classifier from xgboost
    sub[:,index] = clf.predict_proba(test)[:,1]


# In[ ]:

submissions = pd.DataFrame(sub)
submissions['ID'] = pd.Series(test_id)
submissions.to_csv('features15.xgb.probs.csv')

