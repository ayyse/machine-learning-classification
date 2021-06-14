# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 19:19:07 2021

@author: Acer
"""

import pandas as pd
import numpy as np
from sklearn import metrics
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
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

cross_cor_list=list()
cross_cor_and_colname=dict()

# cross correlation calculation function
print("\nAbsoulute Cross-Correlation: \n")
def calculateAbsCrossCor(xname,i):
    x = np.array(preprocessing_data[xname])
    y = np.array(preprocessing_data["target"])
    r=np.abs(np.corrcoef(x,y))
    cross_cor_list.append(r)
    cross_cor_and_colname[xname] = cross_cor_list[0][1]
    print(xname, "\n", r,"\n")

calculateAbsCrossCor("signal_id",1) #column 1
calculateAbsCrossCor("id_measurement",1) #column 2
calculateAbsCrossCor("phase",2)#column 3


def scoreResults(model, X_train, X_test, y_train, y_test):

    y_train_predict = model.predict(X_train)
    y_test_predict = model.predict(X_test)

    r2_train = metrics.r2_score(y_train, y_train_predict)
    r2_test = metrics.r2_score(y_test, y_test_predict)

    mse_train = metrics.mean_squared_error(y_train, y_train_predict)
    mse_test = metrics.mean_squared_error(y_test, y_test_predict)

    return [r2_train, r2_test, mse_train, mse_test, y_train_predict, y_test_predict]

# Makes 10-fold process
def classifierAlgrithm(class_model, X_train, X_test, y_train, y_test, class_name):
    model = class_model
    k = 10
    iter=1
    cv = KFold(n_splits=k, random_state = 0, shuffle=True)
    #print(class_name, "Scores")
    for train_index, test_index in cv.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        model.fit(X_train, y_train)
        result = scoreResults(model = model
                         ,X_train = X_train
                         ,X_test = X_test
                         ,y_train = y_train
                         ,y_test = y_test)
        
    '''print(f"{iter}. veri kesiti")
    print(f"Train R2 Score: {result[0]:.4f} MSE: {result[2]:.4f}")
    print(f"Test R2 Score: {result[1]:9.4f} MSE: {result[3]:.4f}\n")
    iter += 1'''
        
    #Residual Plot
    #plt.scatter(y_train, result[4], color = 'red', label = "Train")
    plt.scatter(y_test, result[5], color = "green", label = "Test")
    plt.hlines(y=0, xmin=-100, xmax=100, color= 'blue')
    plt.title(class_name)
    plt.ylabel('Residual')
    plt.xlabel('Predict Values')
    plt.legend()
    plt.show()
        
    # Curve plot
    #plt.scatter(y_train, result[4], label = "Train", color = "red")
    plt.scatter(y_test, result[5], label = "Test", color = "green")
    plt.plot(range(200), range(200))
    plt.title(class_name)
    plt.ylabel('Target')
    plt.xlabel('Predict Values')
    plt.legend()
    plt.show()
        
    print("------------------------------------------------------------------------")
classifierAlgrithm(classifier_dt, X_train, X_test, y_train, y_test, 'Decision Tree Algorithm')
classifierAlgrithm(classifier_knn, X_train, X_test, y_train, y_test, 'KNN Algorithm')
'''classifierAlgrithm(classifier_lr, X_train, X_test, y_train, y_test, 'Logistic Regression Algorithm')
classifierAlgrithm(classifier_nb, X_train, X_test, y_train, y_test, 'Naive Bayes Algorithm')
classifierAlgrithm(classifier_rf, X_train, X_test, y_train, y_test, 'Random Forest Algorithm')
classifierAlgrithm(classifier_ab, X_train, X_test, y_train, y_test, 'Ada Boost Algorithm')
classifierAlgrithm(classifier_sv, X_train, X_test, y_train, y_test, 'SVC Algorithm')
classifierAlgrithm(classifier_mlp, X_train, X_test, y_train, y_test, 'Multi-Layer Perceptron (MLP/ANN) Algorithm')'''
    
# draw histograms
models = []
models.append(('Decision Tree', classifier_dt))
models.append(('KNN', classifier_knn))
models.append(('Logistic', classifier_lr))
models.append(('Naive Bayes', classifier_nb))
models.append(('Random Forest', classifier_rf))
models.append(('Ada Boost', classifier_ab))
'''models.append(('SVC', classifier_sv))
models.append(('Multi-Layer',classifier_mlp))'''

valuesMSE = []
valuesR2 =[]
names=[]

for name, model in models:
    kfold = KFold(n_splits=10, shuffle=True, random_state=0)
    
    resultsMSE = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    resultsR2 = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
    
    valuesMSE.append(resultsMSE)
    valuesR2.append(resultsR2)
    names.append(name)
    mse = "MSE - %s: %f" % (name, abs(resultsMSE.mean()))
    r2  = "R2 - %s: %f" % (name, abs(resultsR2.mean()))
    print(mse)
    print(r2)

print("\n-------------------------------------------------------------------")
puan=[]
for i in range(len(names)):
    puan.append(abs(valuesMSE[i].mean()))
print("\nBest learning algorithm (Accuracy):")
print(names[puan.index(min(puan))], "->", min(puan))

#Plotting histogram of R2
r2_hist = plt.hist(valuesR2)
plt.title('Histogram of R2 results')
plt.show()

#Plotting histogram of MSE
mse_hist = plt.hist(valuesMSE)
plt.title('Histogram of MSE results')
plt.show()

# residual and curve plot
fig = plt.figure()
fig.suptitle("Algorithm Comparision")

nax=len(models)
i=1
for name, model in models:
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    
    curveaxis=np.zeros((100, X_test.shape[1]))
    for cx in range(X_test.shape[1]):
        curveaxis[:,cx]=np.linspace(np.min(X_test[:,cx]),np.max(X_test[:,cx]),100)  
    curve_predictions = model.predict(curveaxis) 

    plt.subplot(5,3,i)
    plt.title(name)
    plt.scatter(X_test[:,0], y_test,c='b')
    plt.scatter(X_test[:,0], y_test_pred,c='r',alpha=0.5)
    plt.plot(curveaxis[:,0], curve_predictions,c='y')
    plt.grid()
    
    i=i+1 # subplot indeksi

#******************************************************************************
                         #MADDE-5
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
#
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


#Decision Tree Parametreleri için(link = https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
parameters = {'criterion':('gini', 'entropy'), 'splitter':('best', 'random'), 'max_depth':[1,10]}
scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(DecisionTreeClassifier(), parameters, scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

