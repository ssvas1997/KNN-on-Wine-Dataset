# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 09:32:38 2019

@author: mca
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

df= pd.read_csv("E:\Srinivas\KNN\wine.csv")
features= df.drop('class',axis=1).values
classes=df['class'].values
(train_feat,test_feat,train_classes,test_classes)= train_test_split(features,classes,train_size=0.8,random_state=10)

knn= KNeighborsClassifier(n_neighbors=4)
knn.fit(train_feat,train_classes)
pred=knn.predict(test_feat)
print("Accuracy:",metrics.accuracy_score(test_classes,pred))

dft= pd.read_csv("E:\Srinivas\KNN\wine2.csv")
feat=dft[dft.columns].values
pred=knn.predict(feat)
print("Target Class",pred)

neighbors= np.arange(1,9) #k value
train_accuracy= np.empty(len(neighbors))
test_accuracy= np.empty(len(neighbors))
for i,k in enumerate(neighbors):
    #Set up a knn classifier with k neighbors
    knn= KNeighborsClassifier(n_neighbors=k)
    #Fit the model
    knn.fit(train_feat,test_classes)
    #Compute Accuracy on the training set
    train_accuracy[i] = knn.score(train_feat,train_classes)
    #Compute accuracy on the test set
    test_accuracy[i]= knn.score(test_feat,test_classes)
    