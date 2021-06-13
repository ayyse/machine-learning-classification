# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 19:19:07 2021

@author: Acer
"""

import pandas as pd
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score

# ********************* Read Data *******************************************
df = pd.read_excel('C:/Users/Acer/Desktop/metadata_train.xlsx', header = None)
df.drop([0], axis = 0, inplace = True)

X = df.iloc[:,:-1].values
y = df[3].values.astype('int')
# ******************************************************************************

# ********************* Train & Test *******************************************
#separates data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# ******************************************************************************

# ********************* Preprocessing ******************************************
preprocessing_data = pd.DataFrame(StandardScaler().fit(df).transform(df))
# ******************************************************************************

# ********************* Feature Extraction *************************************
def getfeature(data):
    fmean=np.mean(data)
    fstd=np.std(data)
    fmax=np.max(data)
    fmin=np.min(data)
    fkurtosis=scipy.stats.kurtosis(data)
    zero_crosses = np.nonzero(np.diff(data > 0))[0]
    fzero=zero_crosses.size/len(data)
    return fmean,fstd,fmax,fmin,fkurtosis,fzero
def extractFeature(raw_data,ws,hop,dfname):
    fmean=[]
    fstd=[]
    fmax=[]
    fmin=[]
    fkurtosis=[]
    fzero=[]
    flabel=[]
    for i in range(ws,len(raw_data),hop):
       m,s,ma,mi,k,z = getfeature(raw_data.iloc[i-ws+1:i,0])
       fmean.append(m)
       fstd.append(s)
       fmax.append(ma)
       fmin.append(mi)
       fzero.append(z)
       fkurtosis.append(k)
       
       flabel.append(dfname)
    rdf = pd.DataFrame(
    {'mean': fmean,
     'std': fstd,
     'max': fmax,
     'min': fmin,
     'kurtosis': fkurtosis,
     'zerocross':fzero,
     'label':flabel
    })
    return rdf
# not fault train
notfaulttrain = extractFeature(preprocessing_data,250,10,"0")
# fault train
faulttrain = extractFeature(preprocessing_data,250,10,"1")

train = pd.concat([notfaulttrain, faulttrain])

# not fault test
notfaulttest = extractFeature(preprocessing_data,250,10,"0")
# fault test
faulttest = extractFeature(preprocessing_data,250,10,"1")

test = pd.concat([notfaulttest, faulttest])
# ******************************************************************************

# ************************ Decision Tree Algorithm *****************************
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# number of correct and incorrect predictions
cm = confusion_matrix(y_test, y_pred)  

print('Accuracy score on train data:', accuracy_score(y_true = y_train, y_pred = classifier.predict(X_train)))
print('Accuracy score on test data:', accuracy_score(y_true = y_test, y_pred = classifier.predict(X_test)))
# ******************************************************************************




