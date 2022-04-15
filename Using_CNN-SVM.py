# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 10:07:57 2022

@author: sjhan
"""

from sklearn.svm import SVC
from Using_CNN import feat_trainCNN,feat_testCNN,y_train,y_test
import numpy as np

svm = SVC(kernel='linear')
svm.fit(feat_trainCNN,np.argmax(y_train,axis=1))

TrainSVMScoreCNN=svm.score(feat_trainCNN,np.argmax(y_train,axis=1))*100
print("SVM Training Accuracy Score:-",TrainSVMScoreCNN)

TestSVMScoreCNN=svm.score(feat_testCNN,np.argmax(y_test,axis=1))*100
print("\nSVM Testing Accuracy Score:-",TestSVMScoreCNN)
