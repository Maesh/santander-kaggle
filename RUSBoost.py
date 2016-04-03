from sklearn import svm
from sklearn import tree
from sklearn import cross_validation
from sklearn import preprocessing, metrics
from math import log
from dataio import getdata, writesub
import xgboost as xgb

import random
import pandas as pd
import numpy as np
import warnings 

"""
This implementation of RUSBoost was originally written by github user
harusametime, and can be found at:

https://github.com/harusametime/RUSBoost

Edits by Ryan Gooch, Mar 2016, for use in the Kaggle Santander competition.
somewhat heavy altering was needed to incorporate this with the way I 
handle data.
"""

def countvals(a,val) :
    """
    Convenience function to count number of values in numpy array

    a is the array, and val is the numeric value to count.
    """

    return np.max(np.where(a==val)[0].shape)

class RUSBoost:


    def __init__(self, instances, labels, base_classifier, n_classifier, balance):
        
        self.w_update=[]
        self.clf = []
        self.n_classifier = n_classifier
        for i in range(n_classifier):
            self.clf.append(base_classifier)
        self.rate = balance
        self.X = instances
        self.Y = labels
        
        # initialize weight
        self.weight = []
        self.init_w = 1.0/len(self.X)
        for i in range(len(self.X)):
            self.weight.append(self.init_w)
    
    def classify(self, instance):
        
        positive_score = 0 # in case of +1
        negative_score = 0 # in case of 0
        
        for k in range(self.n_classifier):
            if self.clf[k].predict(instance) == 1:
                positive_score += log(1/self.w_update[k])
            else:
                negative_score += log(1/self.w_update[k])
        if negative_score <= positive_score:
            return 1
        else:
            return 0
        
    def learning(self):
        
        k = 0
        while k < self.n_classifier:
            
            sampled = self.undersampling()
            sampled_X = []
            sampled_Y = []
            # from paper, weight is 1/m, where m is number of majority
            # class samples
            sampled_weight = []
            
            for s in sampled:
                sampled_X.append(s[1])
                sampled_Y.append(s[2])
                sampled_weight.append(self.weight[s[0]])
                
            self.clf[k].fit(sampled_X, sampled_Y, sampled_weight)
           
   
            loss = 0
            for i in range(len(self.X)):
                if self.Y[i] == self.clf[k].predict(self.X[i]):
                    continue
                else:
                    loss += self.weight[i]
    
            self.w_update.append(loss/(1-loss))
        
            for i in range(len(self.weight)):
                if loss == 0:
                    self.weight[i] = self.weight[i] #wut
                elif self.Y[i] == self.clf[k].predict(self.X[i]):
                    self.weight[i] = self.weight[i] * (loss / (1 - loss))
                       
            sum_weight = 0
            for i in range(len(self.weight)):
                sum_weight += self.weight[i]
              
            for i in range(len(self.weight)):
                self.weight[i] = self.weight[i] / sum_weight
            k = k + 1
     
            
    def undersampling(self):
        
        '''Check the major class'''
        #diff = self.Y.count(1) > self.Y.count(0)
        diff = countvals(self.Y,1) > countvals(self.Y,0)
        delete_list = []
        keep_list =[]
        if  diff > 0:
            for i in range(len(self.Y)):
                if self.Y[i] == 1:
                    delete_data = [i, self.X[i], 1]
                    delete_list.append(delete_data)
                else:
                    keep_data = [i, self.X[i], 0]
                    keep_list.append(keep_data)
        else:
            for i in range(len(self.Y)):
                if self.Y[i] == 0:
                    delete_data = [i, self.X[i], 0]
                    delete_list.append(delete_data)
                else:
                    keep_data = [i, self.X[i], 1]
                    keep_list.append(keep_data)
        
        while len(keep_list) != self.rate*(len(delete_list)+len(keep_list)):
            k = random.choice(range(len(delete_list)))
            delete_list.pop(k)
        
        all_list = delete_list + keep_list
        return sorted(all_list, key=lambda x:x[2])
    
if __name__ == '__main__':
    
    # Functionality deprecated in 0.19, just ignore for now
    warnings.filterwarnings('ignore')

    ''' 
    Choose a base classifier, e.g. SVM, Decision Tree
    '''
    #base_classifier = svm.SVC()
    base_classifier = tree.DecisionTreeClassifier()
    # base_classifier = xgb.XGBClassifier(missing=np.nan, max_depth=5, 
    #     n_estimators=25, learning_rate=0.03, nthread=4, 
    #     subsample=0.95, colsample_bytree=0.85, seed=19683)

    '''
    Set the number of base classifiers
    ''' 
    N = 100
    
    '''
    Set the rate of minor instances to major instances
    If the rate is 0.5, the numbers of both instances become equal
    by random under sampling. 
    ''' 
    rate = 0.5
    
    
    # '''
    # Preparation of data
    #     "Ecoli data" from
    #     http://www.cs.gsu.edu/~zding/research/benchmark-data.php
    #     Test data: randomly selected one
    #     Supervised data: the others 
    # '''
    # supervisedData =[]
    # supervisedLabel=[]
    # testData =[]
    # testLabel =[]
    # n_features = 7
    
    # # Read from a text file
    # lines = [] ## all lines
    # for line in open('x1data.txt', 'r'):
    #     elements = line[:-1].split(' ')
    #     lines.append(elements)
    
    # # Select a line for test data
    # selectline = lines.pop(random.randint(0, len(lines)))
    
    # # Separate each line into a feature vector and a lebel
    # testLabel.append(int(selectline[0]))
    # testData = [0]*n_features
    # for i in range(1,n_features+1):
    #     pair = selectline[i].split(':')
    #     if pair[0] != '':
    #         testData[int(pair[0])-1] = pair[1]
    
    # for line in lines:
    #     supervisedLabel.append(int(line[0]))
    #     sData = [0]*n_features
    #     for i in range(1,n_features+1):
    #         pair = line[i].split(':')
    #         if pair[0] != '':
    #             sData[int(pair[0])-1] = pair[1]
    #     supervisedData.append(sData)

    print('Getting the data\n')
    # Get the data in, skip header row
    # train = np.genfromtxt('train.csv',delimiter=',',skip_header=1)
    df_train, df_test = getdata()

    # Get target values
    y = df_train['TARGET'].values

    X_train = df_train.drop(['ID','TARGET'], axis=1).values

    # Standardize, ignore numerical warning for now
    X = preprocessing.scale(X_train)

    # Random state for repeatability, split into training and validation sets
    rs = 19683
    X_train, X_val, y_train, y_val = \
        cross_validation.train_test_split(X, y, \
            test_size=0.25, random_state=rs)
    
    rus = RUSBoost(X_train, y_train, base_classifier, N, rate)
    
    print('Training the model\n')
    rus.learning()
    print('Classifying validation set\n')
    pred = []
    for i in range(len(X_val)) :
        pred.append(rus.classify(X_val[i]))
    
    roc = metrics.roc_auc_score(y_val, pred)
    print('ROC: %.4f'%(roc))