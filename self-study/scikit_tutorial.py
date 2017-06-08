# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 17:49:01 2017

@author: hoge
"""

import matplotlib
import seaborn as sns
iris= sns.load_dataset('iris')

sns.set()
sns.pairplot(iris, hue='species', size=1.5)

X_iris= iris.drop('species', axis=1)
y_iris= iris['species']


# implementing gaussian naive bayes on IRIS data
import sklearn
from sklearn.cross_validation import train_test_split
Xtrain, Xtest, ytrain, ytest= train_test_split(X_iris, y_iris, test_size= 0.4, random_state=1)

from sklearn.naive_bayes import GaussianNB
model= GaussianNB()
model.fit(Xtrain, ytrain)
y_model= model.predict(Xtest)

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest, y_model))

from sklearn.decomposition import PCA
model= PCA(n_components=2)
model.fit(X_iris)   # this is unsupervised technique so y is not specified
X_2D= model.transform(X_iris)

iris['PCA1'] = X_2D[:,0]
iris['PCA2'] = X_2D[:,1]
sns.lmplot("PCA1", "PCA2", hue='species', data=iris, fit_reg=False)

from sklearn.mixture import GMM
model= GMM(n_components=3, covariance_type= 'full')
model.fit(X_iris)   # this is unsupervised technique so y is not specified
y_gmm= model.predict(X_iris)

iris['cluster']=y_gmm
sns.lmplot("PCA1", "PCA2", data=iris, hue = 'species', col='cluster', fit_reg=False)



import csv
def loadCsv(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i]=[float(x) for x in dataset[i]]
    return dataset
    
filename = 'pima-indians-diabetes.csv'
dataset = loadCsv(filename)
print('Loaded data file {0} with {1} rows').format(filename, len(dataset) )