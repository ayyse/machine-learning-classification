# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 19:19:07 2021

@author: Acer
"""

import pandas as pd
import numpy as np
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
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


# ********************* Read Data **************************************************************
df = pd.read_excel('C:/Users/Acer/Desktop/metadata_train.xlsx', header = None)
df.drop([0], axis = 0, inplace = True)

X = df.iloc[:,:-1].values
y = df[3].values.astype('int')
# **********************************************************************************************

# ********************* Train & Test ***********************************************************
#separates data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# **********************************************************************************************

# ********************* Preprocessing **********************************************************
preprocessing_data = pd.DataFrame(StandardScaler().fit(df).transform(df))
# **********************************************************************************************

# ********************* Feature Extraction *****************************************************
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
# **********************************************************************************************

# ************************ Classification Algorithms (STEP 3) **********************************
classifier_dt = DecisionTreeClassifier()
classifier_knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )
classifier_lr = LogisticRegression(random_state = 0)
classifier_nb = GaussianNB()  
classifier_rf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0) 
classifier_ab = AdaBoostClassifier(n_estimators=50, learning_rate=1)
classifier_sv = SVC()
classifier_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

def classificationAlgorithms(cls_name, cls_model, X_train, X_test, y_train, y_test):
    model = cls_model
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)


    print(cls_name, 'accuracy score', accuracy_score(y_true = y_train, y_pred = model.predict(X_train)))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred), '\n')
    

classificationAlgorithms('Decision Tree Algorithm', classifier_dt, X_train, X_test, y_train, y_test)
classificationAlgorithms('KNN Algorithm', classifier_knn, X_train, X_test, y_train, y_test)
classificationAlgorithms('Logistic Regression Algorithm', classifier_lr, X_train, X_test, y_train, y_test)
classificationAlgorithms('Naive Bayes Algorithm', classifier_nb, X_train, X_test, y_train, y_test)
classificationAlgorithms('Random Forest Algorithm', classifier_rf, X_train, X_test, y_train, y_test)
classificationAlgorithms('AdaBoost Algorithm', classifier_ab, X_train, X_test, y_train, y_test)
classificationAlgorithms('SVC Algorithm', classifier_sv, X_train, X_test, y_train, y_test)
classificationAlgorithms('MLP Algorithm', classifier_mlp, X_train, X_test, y_train, y_test)
# **************************************************************************************

# ************************ K-FOLD & BOXPLOTS (STEP 4) **********************************
models = []

models.append(classifier_dt)
models.append(classifier_knn)
models.append(classifier_lr)
models.append(classifier_nb)
models.append(classifier_rf)
models.append(classifier_ab)
models.append(classifier_sv)
models.append(classifier_mlp)

names = []

names.append('DT')
names.append('KNN')
names.append('LR')
names.append('NB')
names.append('RF')
names.append('AB')
names.append('SVC')
names.append('MLP')


results = []

for model in models:
    score = cross_val_score(model, X_train, y_train, cv=10)
    results.append(score)
    
puan = []

for i in range(len(names)):
    puan.append(results[i].mean())
print("Highest accuracy value:")
print(names[puan.index(max(puan))], max(puan))  


ax = sns.boxplot(data = results)
ax.set_xticklabels(names)
plt.show()
# ***************************************************************************************




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

