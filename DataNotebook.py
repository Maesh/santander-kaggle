
# coding: utf-8

# In[1]:

# Based on https://www.kaggle.com/benhamner/d/uciml/iris/python-data-visualizations/notebook
# First, we'll import pandas, a data processing and CSV file I/O library
import pandas as pd
import numpy as np

# We'll also import seaborn, a Python graphing library
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

# Next, we'll load the train and test dataset, which is in the "../input/" directory
train = pd.read_csv("train.csv") # the train dataset is now a Pandas DataFrame
test = pd.read_csv("test.csv") # the test dataset is now a Pandas DataFrame

get_ipython().magic(u'pylab inline')
# Let's see what's in the trainings data - Jupyter notebooks print the result of the last thing you do
train.head()


# In[2]:

# 116 values in column var3 are -999999
# var3 is suspected to be the nationality of the customer
# -999999 would mean that the nationality of the customer is unknown
train.loc[train.var3==-999999].shape


# In[3]:

# Replace -999999 in var3 column with most common value 2 
# See https://www.kaggle.com/cast42/santander-customer-satisfaction/debugging-var3-999999
# for details
train = train.replace(-999999,2)
train.loc[train.var3==-999999].shape


# In[4]:

# var38 is important according to XGBOOST
# see https://www.kaggle.com/cast42/santander-customer-satisfaction/xgboost-with-early-stopping/files
# Also RFC thinks var38 is important
# see https://www.kaggle.com/tks0123456789/santander-customer-satisfaction/data-exploration/notebook
# so far I have not seen a guess what var38 may be about
train.var38.describe()


# In[5]:

# How is var38 looking when customer is unhappy ?
train.loc[train['TARGET']==1, 'var38'].describe()


# In[6]:

# Histogram for var 38 is not normal distributed
train.var38.hist(bins=1000)


# In[7]:

train.var38.map(np.log).hist(bins=1000)


# In[8]:

# where is the spike between 11 and 12  in the log plot ?
train.var38.map(np.log).mode()


# In[9]:

# What are the most common values for var38 ?
train.var38.value_counts()


# In[10]:

# what is we exclude the most common value
train.loc[~np.isclose(train.var38, 117310.979016), 'var38'].value_counts()


# In[11]:

# Look at the distribution
train.loc[~np.isclose(train.var38, 117310.979016), 'var38'].map(np.log).hist(bins=100)


# In[12]:

# Above plot suggest we split up var38 into two variables
# var38mc == 1 when var38 has the most common value and 0 otherwise
# logvar38 is log transformed feature when var38mc is 0, zero otherwise
train['var38mc'] = np.isclose(train.var38, 117310.979016)
train['logvar38'] = train.loc[~train['var38mc'], 'var38'].map(np.log)
train.loc[train['var38mc'], 'logvar38'] = 0


# In[13]:

#Check for nan's
print('Number of nan in var38mc', train['var38mc'].isnull().sum())
print('Number of nan in logvar38',train['logvar38'].isnull().sum())


# In[14]:

train['var15'].describe()


# In[15]:

#Looks more normal, plot the histogram
train['var15'].hist(bins=100)


# In[16]:

# Let's look at the density of the age of happy/unhappy customers
sns.FacetGrid(train, hue="TARGET", size=6)    .map(sns.kdeplot, "var15")    .add_legend()
plt.title('Unhappy customers are slightly older')
plt.show()


# In[17]:

sns.FacetGrid(train, hue="TARGET", size=10)    .map(plt.scatter, "var38", "var15")    .add_legend()


# In[18]:

sns.FacetGrid(train, hue="TARGET", size=10)    .map(plt.scatter, "logvar38", "var15")    .add_legend()
plt.ylim([0,120]) # Age must be positive ;-)


# In[19]:

# Exclude most common value for var38 
sns.FacetGrid(train[~train.var38mc], hue="TARGET", size=10)    .map(plt.scatter, "logvar38", "var15")    .add_legend()
plt.ylim([0,120])


# In[20]:

# What is distribution of the age when var38 has it's most common value ?
sns.FacetGrid(train[train.var38mc], hue="TARGET", size=6)    .map(sns.kdeplot, "var15")    .add_legend()


# In[21]:

X = train.iloc[:,:-1]
y = train.TARGET

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif,chi2
from sklearn.preprocessing import Binarizer, scale

# First select features based on chi2 and f_classif
p = 3

X_bin = Binarizer().fit_transform(scale(X))
selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, y)
selectF_classif = SelectPercentile(f_classif, percentile=p).fit(X, y)

chi2_selected = selectChi2.get_support()
chi2_selected_features = [ f for i,f in enumerate(X.columns) if chi2_selected[i]]
print('Chi2 selected {} features {}.'.format(chi2_selected.sum(),
   chi2_selected_features))
f_classif_selected = selectF_classif.get_support()
f_classif_selected_features = [ f for i,f in enumerate(X.columns) if f_classif_selected[i]]
print('F_classif selected {} features {}.'.format(f_classif_selected.sum(),
   f_classif_selected_features))
selected = chi2_selected & f_classif_selected
print('Chi2 & F_classif selected {} features'.format(selected.sum()))
features = [ f for f,s in zip(X.columns, selected) if s]
print (features)


# In[22]:

# Make a dataframe with the selected features and the target variable
X_sel = train[features+['TARGET']]


# In[23]:

# var38 (important for XGB and RFC is not selected but var36 is. Let's explore
X_sel['var36'].value_counts()


# In[24]:

# Let's plot the density in function of the target variabele
sns.FacetGrid(train, hue="TARGET", size=6)    .map(sns.kdeplot, "var36")    .add_legend()


# In[25]:

# var36 in function of var38 (most common value excluded) 
sns.FacetGrid(train[~train.var38mc], hue="TARGET", size=10)    .map(plt.scatter, "var36", "logvar38")    .add_legend()


# In[26]:

sns.FacetGrid(train[(~train.var38mc) & (train.var36 < 4)], hue="TARGET", size=10)    .map(plt.scatter, "var36", "logvar38")    .add_legend()


# In[27]:

# Let's plot the density in function of the target variabele, when var36 = 99
sns.FacetGrid(train[(~train.var38mc) & (train.var36 ==99)], hue="TARGET", size=6)    .map(sns.kdeplot, "logvar38")    .add_legend()


# In[28]:

sns.pairplot(train[['var15','var36','logvar38','TARGET']], hue="TARGET", size=2, diag_kind="kde")


# In[29]:

train[['var15','var36','logvar38','TARGET']].boxplot(by="TARGET", figsize=(12, 6))


# In[30]:

# A final multivariate visualization technique pandas has is radviz
# Which puts each feature as a point on a 2D plane, and then simulates
# having each sample attached to those points through a spring weighted
# by the relative value for that feature
from pandas.tools.plotting import radviz
radviz(train[['var15','var36','logvar38','TARGET']], "TARGET")


# In[31]:

features+['TARGET']


# In[32]:

radviz(train[features], "TARGET")


# In[33]:

# this compares features pairwise and looks at the distribution of said features for each target value, 1 and 0
sns.pairplot(train[features], hue="TARGET", size=2, diag_kind="kde")


# In[31]:

train[features]


# In[35]:

# final check for NaNs
train[features].isnull().sum()


# In[36]:

test = pd.read_csv("test.csv") # the test dataset is now a Pandas DataFrame


# # Combining Features Efficiently
# Some machine learning techniques, such as Neural Networks, can combine features intelligently on their own. With this competition, however, the class imbalance is apparently rendering NNs difficult to use, so we can instead try to combine them ourselves. The following is taken from a script: https://www.kaggle.com/selfishgene/santander-customer-satisfaction/advanced-feature-exploration

# In[4]:

from sklearn import cluster
from sklearn import ensemble
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score as auc
import time

plt.rcParams['figure.figsize'] = (10, 10)
rs = 19683
#%% load data and remove constant and duplicate columns  (taken from a kaggle script)

trainDataFrame = pd.read_csv('train.csv')

# remove constant columns
colsToRemove = []
for col in trainDataFrame.columns:
    if trainDataFrame[col].std() == 0:
        colsToRemove.append(col)

trainDataFrame.drop(colsToRemove, axis=1, inplace=True)

# remove duplicate columns
colsToRemove = []
columns = trainDataFrame.columns
for i in range(len(columns)-1):
    v = trainDataFrame[columns[i]].values
    for j in range(i+1,len(columns)):
        if np.array_equal(v,trainDataFrame[columns[j]].values):
            colsToRemove.append(columns[j])

trainDataFrame.drop(colsToRemove, axis=1, inplace=True)

trainLabels = trainDataFrame['TARGET']
trainFeatures = trainDataFrame.drop(['ID','TARGET'], axis=1)


#%% look at single feature performance

X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(trainFeatures, trainLabels, 
                                                                       test_size=0.5, random_state=rs)
verySimpleLearner = ensemble.GradientBoostingClassifier(n_estimators=20, max_features=1, max_depth=3, 
                                                        min_samples_leaf=100, learning_rate=0.1, 
                                                        subsample=0.65, loss='deviance', random_state=rs)

startTime = time.time()
singleFeatureTable = pd.DataFrame(index=range(len(X_train.columns)), columns=['feature','AUC'])
for k,feature in enumerate(X_train.columns):
    trainInputFeature = X_train[feature].values.reshape(-1,1)
    validInputFeature = X_valid[feature].values.reshape(-1,1)
    verySimpleLearner.fit(trainInputFeature, y_train)
    
    validAUC = auc(y_valid, verySimpleLearner.predict_proba(validInputFeature)[:,1])
    singleFeatureTable.ix[k,'feature'] = feature
    singleFeatureTable.ix[k,'AUC'] = validAUC
        
print("finished evaluating single features. took %.2f minutes" %((time.time()-startTime)/60))


# We can show the single feature Area Under Curve Performance

# In[5]:

#%% sort according to AUC and present the table
singleFeatureTable = singleFeatureTable.sort_values(by='AUC', axis=0, ascending=False).reset_index(drop=True)

singleFeatureTable.ix[:15,:]


# In[6]:

#%% find interesting fivewise combinations

numFeaturesInCombination = 5
numCombinations = 5000
numBestSingleFeaturesToSelectFrom = 50

X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(trainFeatures, trainLabels, 
                                                                       test_size=0.5, random_state=rs)
weakLearner = ensemble.GradientBoostingClassifier(n_estimators=50, max_features=2, max_depth=6, 
                                                  min_samples_leaf=100,learning_rate=0.1, 
                                                  subsample=0.65, loss='deviance', random_state=rs)

featuresToUse = singleFeatureTable.ix[0:numBestSingleFeaturesToSelectFrom-1,'feature']
featureColumnNames = ['feature'+str(x+1) for x in range(numFeaturesInCombination)]
featureCombinationsTable = pd.DataFrame(index=range(numCombinations), columns=featureColumnNames + ['combinedAUC'])

# for numCombinations iterations 
startTime = time.time()
for combination in range(numCombinations):
    # generate random feature combination
    randomSelectionOfFeatures = sorted(np.random.choice(len(featuresToUse), numFeaturesInCombination, replace=False))

    # store the feature names
    combinationFeatureNames = [featuresToUse[x] for x in randomSelectionOfFeatures]
    for i in range(len(randomSelectionOfFeatures)):
        featureCombinationsTable.ix[combination,featureColumnNames[i]] = combinationFeatureNames[i]

    # build features matrix to get the combination AUC
    trainInputFeatures = X_train.ix[:,combinationFeatureNames]
    validInputFeatures = X_valid.ix[:,combinationFeatureNames]
    # train learner
    weakLearner.fit(trainInputFeatures, y_train)
    # store AUC results
    validAUC = auc(y_valid, weakLearner.predict_proba(validInputFeatures)[:,1])        
    featureCombinationsTable.ix[combination,'combinedAUC'] = validAUC

validAUC = np.array(featureCombinationsTable.ix[:,'combinedAUC'])
print("(min,max) AUC = (%.4f,%.4f). took %.1f minutes" % (validAUC.min(),validAUC.max(), (time.time()-startTime)/60))

# show the histogram of the feature combinations performance 
plt.figure(); plt.hist(validAUC, 100, facecolor='blue', alpha=0.75)
plt.xlabel('AUC'); plt.ylabel('frequency'); plt.title('feature combination AUC histogram'); plt.show()


# In[7]:

#%% sort according to combination AUC and look at the table

featureCombinationsTable = featureCombinationsTable.sort_values(by='combinedAUC', axis=0, ascending=False).reset_index(drop=True)
featureCombinationsTable.ix[:20,:]


# In[8]:

featureCombinationsTable.to_csv('feature_combos_5.csv',',')


# In[ ]:

#%% find interesting fivewise combinations

numFeaturesInCombination = 15
numCombinations = 10000
numBestSingleFeaturesToSelectFrom = 50

X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(trainFeatures, trainLabels, 
                                                                       test_size=0.5, random_state=rs)
weakLearner = ensemble.GradientBoostingClassifier(n_estimators=50, max_features=2, max_depth=6, 
                                                  min_samples_leaf=100,learning_rate=0.1, 
                                                  subsample=0.65, loss='deviance', random_state=rs)

featuresToUse = singleFeatureTable.ix[0:numBestSingleFeaturesToSelectFrom-1,'feature']
featureColumnNames = ['feature'+str(x+1) for x in range(numFeaturesInCombination)]
featureCombinationsTable2 = pd.DataFrame(index=range(numCombinations), columns=featureColumnNames + ['combinedAUC'])

# for numCombinations iterations 
startTime = time.time()
for combination in range(numCombinations):
    # generate random feature combination
    randomSelectionOfFeatures = sorted(np.random.choice(len(featuresToUse), numFeaturesInCombination, replace=False))

    # store the feature names
    combinationFeatureNames = [featuresToUse[x] for x in randomSelectionOfFeatures]
    for i in range(len(randomSelectionOfFeatures)):
        featureCombinationsTable2.ix[combination,featureColumnNames[i]] = combinationFeatureNames[i]

    # build features matrix to get the combination AUC
    trainInputFeatures = X_train.ix[:,combinationFeatureNames]
    validInputFeatures = X_valid.ix[:,combinationFeatureNames]
    # train learner
    weakLearner.fit(trainInputFeatures, y_train)
    # store AUC results
    validAUC = auc(y_valid, weakLearner.predict_proba(validInputFeatures)[:,1])        
    featureCombinationsTable2.ix[combination,'combinedAUC'] = validAUC

validAUC = np.array(featureCombinationsTable2.ix[:,'combinedAUC'])
print("(min,max) AUC = (%.4f,%.4f). took %.1f minutes" % (validAUC.min(),validAUC.max(), (time.time()-startTime)/60))

# show the histogram of the feature combinations performance 
plt.figure(); plt.hist(validAUC, 100, facecolor='blue', alpha=0.75)
plt.xlabel('AUC'); plt.ylabel('frequency'); plt.title('feature combination AUC histogram'); plt.show()


# In[ ]:

#%% sort according to combination AUC and look at the table

featureCombinationsTable2 = featureCombinationsTable2.sort_values(by='combinedAUC', axis=0, ascending=False).reset_index(drop=True)
featureCombinationsTable2.ix[:20,:]


# In[ ]:

featureCombinationsTable2.to_csv('feature_combos_15.csv',',')


# In[ ]:

#%% find interesting fivewise combinations

numFeaturesInCombination = 30
numCombinations = 10000
numBestSingleFeaturesToSelectFrom = 50

X_train, X_valid, y_train, y_valid = cross_validation.train_test_split(trainFeatures, trainLabels, 
                                                                       test_size=0.5, random_state=rs)
weakLearner = ensemble.GradientBoostingClassifier(n_estimators=50, max_features=2, max_depth=6, 
                                                  min_samples_leaf=100,learning_rate=0.1, 
                                                  subsample=0.65, loss='deviance', random_state=rs)

featuresToUse = singleFeatureTable.ix[0:numBestSingleFeaturesToSelectFrom-1,'feature']
featureColumnNames = ['feature'+str(x+1) for x in range(numFeaturesInCombination)]
featureCombinationsTable3 = pd.DataFrame(index=range(numCombinations), columns=featureColumnNames + ['combinedAUC'])

# for numCombinations iterations 
startTime = time.time()
for combination in range(numCombinations):
    # generate random feature combination
    randomSelectionOfFeatures = sorted(np.random.choice(len(featuresToUse), numFeaturesInCombination, replace=False))

    # store the feature names
    combinationFeatureNames = [featuresToUse[x] for x in randomSelectionOfFeatures]
    for i in range(len(randomSelectionOfFeatures)):
        featureCombinationsTable3.ix[combination,featureColumnNames[i]] = combinationFeatureNames[i]

    # build features matrix to get the combination AUC
    trainInputFeatures = X_train.ix[:,combinationFeatureNames]
    validInputFeatures = X_valid.ix[:,combinationFeatureNames]
    # train learner
    weakLearner.fit(trainInputFeatures, y_train)
    # store AUC results
    validAUC = auc(y_valid, weakLearner.predict_proba(validInputFeatures)[:,1])        
    featureCombinationsTable3.ix[combination,'combinedAUC'] = validAUC

validAUC = np.array(featureCombinationsTable3.ix[:,'combinedAUC'])
print("(min,max) AUC = (%.4f,%.4f). took %.1f minutes" % (validAUC.min(),validAUC.max(), (time.time()-startTime)/60))

# show the histogram of the feature combinations performance 
plt.figure(); plt.hist(validAUC, 100, facecolor='blue', alpha=0.75)
plt.xlabel('AUC'); plt.ylabel('frequency'); plt.title('feature combination AUC histogram'); plt.show()


# In[ ]:

#%% sort according to combination AUC and look at the table

featureCombinationsTable3 = featureCombinationsTable3.sort_values(by='combinedAUC', axis=0, ascending=False).reset_index(drop=True)
featureCombinationsTable3.ix[:20,:]


# In[ ]:

featureCombinationsTable3.to_csv('feature_combos_30.csv',',')


# In[ ]:



