# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 16:37:43 2017

@author: shreya

Reference: https://www.datacamp.com/community/tutorials/machine-learning-python#gs.NeE=TmU
"""

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

digits= datasets.load_digits()   #or use pd.read_csv('source', header=" ")

#exploring data
print(digits.keys())
print(digits.values)
print(digits.data)
print(digits.target)  #OR digits['target']
print(digits.DESCR)


print(digits.data.shape)
print(digits.target.shape)

number_digits= len(np.unique(digits.target))
digit_images = digits.images
print(digit_images.shape)

print(np.all(digits.images.reshape((1797,64))==digits.data))

#plotting images
fig= plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(0,64):
    ax= fig.add_subplot(8,8,i+1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap= plt.cm.binary, interpolation = 'nearest')
    ax.text(0,7,str(digits.target[i]))

plt.show()

#vizualizing your data using dimensionality reduction
from sklearn.decomposition import PCA, RandomizedPCA

randomized_pca= RandomizedPCA(n_components=2)     #decide the no. of pc's using explained variance ratio
reduced_data_rpca= randomized_pca.fit_transform(digits.data)
pca= PCA(n_components=2)
reduced_data_pca= pca.fit_transform(digits.data)     #pca is a linear dimensionality reduction method
print(reduced_data_pca.shape)
colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
for i in range(len(colors)):
    pc1= reduced_data_rpca[:,0][digits.target==i]
    pc2= reduced_data_rpca[:,1][digits.target==i]
    plt.scatter(pc1, pc2 , c= colors[i])
plt.legend(digits.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title("PCA Scatter Plot")
plt.show()

#pre processing of dataset
from sklearn.preprocessing import scale
data= scale(digits.data)

#splitting the data into test and train
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(data, digits.target, digits.images, test_size= 0.25, random_state=42)
n_samples, n_features= X_train.shape
print(X_train.shape)
n_digits= len(np.unique(y_train))
print(len(y_train))

#applying machine learning method: k means clustering
from sklearn.cluster import KMeans
clf= KMeans(init='k-means++', n_clusters=10, random_state=42)  #here clusters=10 because the labels are already known, otherwise AIC and BIC criteria can be used
clf.fit(X_train)# n_init parameter

# vizualize the cluster center images(centroid images)
fig = plt.figure(figsize=(8, 3))
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')
for i in range(10):
    ax = fig.add_subplot(2, 5, 1 + i)
    ax.imshow(clf.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
    plt.axis('off')
plt.show()

#vizualize the data clusters and their centres




#apply prediction
y_pred=clf.predict(X_test)
print(clf.cluster_centres_.shape)

#vizualization of result
from sklearn.manifold import Isomap
X_iso= Isomap(n_neighbors=10).fit_transform(X_train)    #Isomap is a non-linear dimensionality reduction method  #try with pca also and see the difference
print(X_iso.shape)
clusters= clf.fit_predict(X_train)
fig, ax= plt.subplots (1,2, figsize=(8,4))
fig.suptitle('Predicted Versus Training Labels', fontsize=14, fontweight='bold')
fig.subplots_adjust(top=0.85)
ax[0].scatter(X_iso[:, 0], X_iso[:, 1], c=clusters)
ax[0].set_title('Predicted Training Labels')
ax[1].scatter(X_iso[:, 0], X_iso[:, 1], c=y_train)
ax[1].set_title('Actual Training Labels')


#model evaluation
from sklearn import metrics
print(metrics.confusion_matrix(y_test, y_pred))
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
print('% 9s' % 'inertia    homo   compl  v-meas     ARI AMI  silhouette')
print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          %(clf.inertia_,
      homogeneity_score(y_test, y_pred),
      completeness_score(y_test, y_pred),
      v_measure_score(y_test, y_pred),
      adjusted_rand_score(y_test, y_pred),
      adjusted_mutual_info_score(y_test, y_pred),
      silhouette_score(X_test, y_pred, metric='euclidean')))

#try another model: SVM

X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(digits.data, digits.target, digits.images, test_size=0.25, random_state=42)

from sklearn import svm

svc_model= svm.SVC(gamma=0.001, C=100., kernel= 'linear')
svc_model.fit(X_train, y_train)

#parameter tuning using Gridsearch
from sklearn.grid_search import GridSearchCV
parameter_candidates=[{'C':[1,10,100,1000], 'kernel':['linear']},
                       {'C':[1,10,100,1000], 'gamma':[0.001,0.0001],'kernel':['rbf']},]

clf= GridSearchCV(estimator= svm.SVC(), param_grid= parameter_candidates, n_jobs=-1)
clf.fit(X_train, y_train)
print('Best score for training data:', clf.best_score_)
print('Best `C`:',clf.best_estimator_.C)
print('Best kernel:',clf.best_estimator_.kernel)
print('Best `gamma`:',clf.best_estimator_.gamma)
clf.score(X_test, y_test)

#using the best parameters to train the svc model

print(svm.SVC(C=10, kernel= 'rbf', gamma= 0.001).fit(X_train,y_train).score(X_test, y_test))

#using the old svc model
#predict
y_pred= svc_model.predict(X_test)

#vizualizing the rsults
predicted= svc_model.predict(X_test)

images_and_predictions= list(zip(images_test, predicted))

for index, (image,prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(1,4,index+1)
    plt.axis('off')
    plt.imshow(image, cmap= plt.cm.gray_r, interpolation='nearest')
    plt.title('Predicted: ' + str(prediction))
plt.show()    


X_iso = Isomap(n_neighbors=10).fit_transform(X_train)
predicted = svc_model.predict(X_train)
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
fig.subplots_adjust(top=0.85)
ax[0].scatter(X_iso[:, 0], X_iso[:, 1], c=predicted)
ax[0].set_title('Predicted labels')
ax[1].scatter(X_iso[:, 0], X_iso[:, 1], c=y_train)
ax[1].set_title('Actual Labels')
fig.suptitle('Predicted versus actual labels', fontsize=14, fontweight='bold')
plt.show()

#model evaluation
print(metrics.classification_report(y_test, predicted))    
print(metrics.confusion_matrix(y_test, predicted))    
        
    
    
    
    
    
    
    
    
    
    









