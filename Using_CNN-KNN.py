# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 12:04:17 2022

@author: sjhan
"""

from sklearn.neighbors import KNeighborsClassifier
from Using_CNN import feat_trainCNN,feat_testCNN,y_train,y_test
import numpy as np

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(feat_trainCNN,np.argmax(y_train,axis=1))

TrainKNNScoreCNN=knn.score(feat_trainCNN,np.argmax(y_train,axis=1))*100
print("KNN Training Accuracy Score:-",TrainKNNScoreCNN)

TestKNNScoreCNN=knn.score(feat_testCNN,np.argmax(y_test,axis=1))*100
print("\nKNN Testing Accuracy Score:-",TestKNNScoreCNN)
