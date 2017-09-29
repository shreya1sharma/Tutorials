# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:50:21 2017

@author: hoge
"""



import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA, RandomizedPCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import numpy as np

#%%
#read data

header= ['Length', 'area', 'sc', 'compactness', 'r_1_0', 'mean', 'variance', 'class']
df= pd.read_csv('D:\Codes\Ship_Detection\Experiments\Baseline_method\work_data\selected_big\\tankers\\vertical\\features_final.csv', usecols= header)
Y= df[['class']]
X=df.drop('class', axis=1)

print(X.shape)
print(Y.shape)

#standardizing data
min_max_scaler = preprocessing.StandardScaler()
X = min_max_scaler.fit_transform(X.values)
#%%
#vizualize features


pca= PCA(n_components=2)
reduced_X= pca.fit_transform(X)
colours=["blue","red"]
for i in range(len(colours)):
    pc1= reduced_X[:,0][df['class']==i]
    pc2= reduced_X[:,1][df['class']==i]
    plt.scatter(pc1, pc2 , c= colours[i])
    
plt.legend(['cargo','tanker'], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title("PCA Scatter Plot")
plt.show()
#%%
#splitting into test and train
X_train, X_test, Y_train, Y_test= train_test_split(X ,Y, test_size= 0.25, random_state=42)
print(Y_train['class'].value_counts())   #prints the number of samples of each class in training data

#%%
#apply machine learning  method: SVM
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold

clf= svm.SVC(kernel='linear', C=10.0)
clf.fit(X_train,Y_train)
kfold = KFold(n_splits=10, random_state=42)
scores = cross_val_score(clf, X_train, Y_train['class'].ravel(), cv=kfold)   #splitting the training set into training and validation set
#for i in range(3,10):
#    scores = cross_val_score(clf, X_train, Y_train['class'].ravel(), cv=i)
#    predicted=cross_val_predict(clf, X_train, Y_train['class'].ravel(), cv=i)
#    mean_scores.append(metrics.accuracy_score(Y_train['class'],predicted))
#    #metrics.confusion_matrix(Y_test['class'],predicted))
#    
#    
#cvs= np.arange(3,10)
#plt.plot(cvs, mean_scores)
Y_pred= clf.predict(X_test)
    
#%%
#vizualize results on training data

clusters= clf.predict(X_train)
reduced_X_train= PCA(n_components=2).fit_transform(X_train)
fig, ax= plt.subplots(1,2, figsize=(8,4))
fig.suptitle('Predicted versus Actual Labels', fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.85)
ax[0].scatter(reduced_X_train[:, 0], reduced_X_train[:, 1], c=clusters)
ax[0].set_title('Predicted training labels')
ax[1].scatter(reduced_X_train[:, 0], reduced_X_train[:, 1], c=Y_train)
ax[1].set_title('Actual training labels')

print(metrics.accuracy_score(clusters,Y_train))
print(metrics.classification_report(clusters,Y_train))   
print(metrics.confusion_matrix(clusters,Y_train)) 

#%%
#vizualize results on testing data
clusters= clf.predict(X_test)
reduced_X_train= PCA(n_components=2).fit_transform(X_test)
fig, ax= plt.subplots(1,2, figsize=(8,4))
fig.suptitle('Predicted versus Actual Labels', fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.85)
ax[0].scatter(reduced_X_train[:, 0], reduced_X_train[:, 1], c=clusters)
ax[0].set_title('Predicted testing labels')
ax[1].scatter(reduced_X_train[:, 0], reduced_X_train[:, 1], c=Y_test)
ax[1].set_title('Actual testing labels')

#%% Performance evaluation
print(metrics.accuracy_score(Y_test,Y_pred))
print(metrics.classification_report(Y_test,Y_pred))   
print(metrics.confusion_matrix(Y_test,Y_pred))   

#%% Tuning Model parameters for best model selection

parameter_candidates=[{'C':[1,10,15,100,1000], 'kernel':['linear']},
                       {'C':[1,10,100,1000], 'gamma':[0.1, 0.001,0.0001],'kernel':['rbf']}]
clf= GridSearchCV(estimator=svm.SVC(), param_grid= parameter_candidates, scoring='accuracy', cv=8)  #by default cv is 3
clf.fit(X_train, Y_train['class'])
a= clf.cv_results_
means= a['mean_test_score']
stds= a['std_test_score']

for mean, std, params in zip(means, stds, a['params']):
    print("%2f (+/-%2f) for %r"%(mean , std*2, params))

print('Best score for training data:', clf.best_score_)
print('Best parameters:',clf.best_params_)

#print(clf.score(X_test, Y_test['class'].ravel()))
#print(svm.SVC(C=1, kernel= 'rbf', gamma= 0.001).fit(X_train,Y_train['class'].ravel()).score(X_test, Y_test['class'].ravel()))

#clf= svm.SVC(C=1, kernel= 'rbf', gamma= 0.001)
#clf.fit(X_train,Y_train['class'])
clusters= clf.predict(X_train)
print(metrics.accuracy_score(clusters,Y_train))
print(metrics.classification_report(clusters,Y_train))   
print(metrics.confusion_matrix(clusters,Y_train)) 

reduced_X_train= PCA(n_components=2).fit_transform(X_train)
fig, ax= plt.subplots(1,2, figsize=(8,4))
fig.suptitle('Predicted versus Actual Labels', fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.85)
ax[0].scatter(reduced_X_train[:, 0], reduced_X_train[:, 1], c=clusters)
ax[0].set_title('Predicted training labels')
ax[1].scatter(reduced_X_train[:, 0], reduced_X_train[:, 1], c=Y_train)
ax[1].set_title('Actual training labels')

Y_pred= clf.predict(X_test)
print(metrics.accuracy_score(Y_test,Y_pred))
print(metrics.classification_report(Y_test,Y_pred))   
print(metrics.confusion_matrix(Y_test,Y_pred))   

#%%-----------------------------------------------
#creating a graph between cv and score

accuracy=[]

for i in range(3, 10):

    parameter_candidates=[{'C':[1,10,100,1000], 'kernel':['linear']},
                       {'C':[1,10,100,1000], 'gamma':[0.1, 0.001,0.0001],'kernel':['rbf']}]

    clf= GridSearchCV(estimator=svm.SVC(), param_grid= parameter_candidates, scoring='accuracy', cv=i)  #by default cv is 3
    clf.fit(X_train, Y_train['class'])
    clusters= clf.predict(X_train)
    accuracy.append(metrics.accuracy_score(clusters,Y_train))
    

cvs= np.arange(3,10)

plt.plot(cvs, accuracy)










