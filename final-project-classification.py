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
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


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

merged_train = pd.concat([notfaulttrain, faulttrain])

# not fault test
notfaulttest = extractFeature(preprocessing_data,250,10,"0")
# fault test
faulttest = extractFeature(preprocessing_data,250,10,"1")

merged_test = pd.concat([notfaulttest, faulttest])
# ******************************************************************************

# ************************ Decision Tree Algorithm *****************************
classifier_dt = DecisionTreeClassifier()
classifier_dt.fit(X_train, y_train)

y_pred = classifier_dt.predict(X_test)

# number of correct and incorrect predictions
cm_dt = confusion_matrix(y_test, y_pred)  

print('(Decision Tree) Accuracy score on train data:', accuracy_score(y_true = y_train, y_pred = classifier_dt.predict(X_train)))
# print('Accuracy score on test data:', accuracy_score(y_true = y_test, y_pred = classifier.predict(X_test)))
# ******************************************************************************

# ******************************* KNN Algorithm ******************************** 
classifier_knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )  
classifier_knn.fit(X_train, y_train) 

y_pred = classifier_knn.predict(X_test)

cm_knn = confusion_matrix(y_test, y_pred)  

print('(KNN) Accuracy score on train data:', accuracy_score(y_true = y_train, y_pred = classifier_knn.predict(X_train)))
# ******************************************************************************

# ************************* Logistic Regression Algorithm **********************
classifier_lr = LogisticRegression(random_state = 0)  
classifier_lr.fit(X_train, y_train) 

y_pred = classifier_lr.predict(X_test)

cm_lr = confusion_matrix(y_test, y_pred)  

print('(Logistic Regression) Accuracy score on train data:', accuracy_score(y_true = y_train, y_pred = classifier_lr.predict(X_train)))
# ******************************************************************************

# ****************************** Naive Bayes Algorithm *************************
classifier_nb = GaussianNB()  
classifier_nb.fit(X_train, y_train) 

y_pred = classifier_nb.predict(X_test)

cm_nb = confusion_matrix(y_test, y_pred)  

print('(Naive Bayes) Accuracy score on train data:', accuracy_score(y_true = y_train, y_pred = classifier_nb.predict(X_train)))
# ******************************************************************************


# ************************* Random Forest Algorithm ****************************
classifier_rf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0) 
classifier_rf.fit(X_train, y_train) 

y_pred = classifier_rf.predict(X_test)

cm_rf = confusion_matrix(y_test, y_pred)  

print('(Random Forest) Accuracy score on train data:', accuracy_score(y_true = y_train, y_pred = classifier_rf.predict(X_train)))
# ******************************************************************************


# ************************* AdaBoost Algorithm ****************************
classifier_ab = AdaBoostClassifier(n_estimators=50, learning_rate=1)
classifier_ab.fit(X_train, y_train) 

y_pred = classifier_ab.predict(X_test)

cm_ab = confusion_matrix(y_test, y_pred)  

print('(AdaBoost) Accuracy score on train data:', accuracy_score(y_true = y_train, y_pred = classifier_ab.predict(X_train)))
# ******************************************************************************


# ****************************** SVC Algorithm *********************************
classifier_sv = SVC(kernel='linear', random_state = 0)
classifier_sv.fit(X_train, y_train) 

y_pred = classifier_sv.predict(X_test)

cm_sv = confusion_matrix(y_test, y_pred)  

print('(SVC) Accuracy score on train data:', accuracy_score(y_true = y_train, y_pred = classifier_sv.predict(X_train)))
# ******************************************************************************


# ***************** Multi-Layer Perceptron (MLP/ANN) Algorithm *****************
classifier_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
classifier_mlp.fit(X_train, y_train) 

y_pred = classifier_mlp.predict(X_test)

cm_mlp = confusion_matrix(y_test, y_pred)  

print('(MLP) Accuracy score on train data:', accuracy_score(y_true = y_train, y_pred = classifier_mlp.predict(X_train)))
# ******************************************************************************



