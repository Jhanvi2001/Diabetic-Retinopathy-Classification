# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 12:47:21 2022

@author: sjhan
"""

from sklearn.naive_bayes import GaussianNB
from Using_CNN import feat_trainCNN,feat_testCNN,y_train,y_test
import numpy as np

gnb = GaussianNB()
gnb.fit(feat_trainCNN,np.argmax(y_train,axis=1))

TrainNBScoreCNN=gnb.score(feat_trainCNN,np.argmax(y_train,axis=1))*100
print("\nGaussianNaive Bayes Training Accuracy Score:-",TrainNBScoreCNN)

TestNBScoreCNN=gnb.score(feat_testCNN,np.argmax(y_test,axis=1))*100
print("\nGaussianNaive Bayes Testing Accuracy Score:-",TestNBScoreCNN)