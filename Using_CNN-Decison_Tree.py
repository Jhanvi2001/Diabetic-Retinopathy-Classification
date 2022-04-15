# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 12:41:09 2022

@author: sjhan
"""

from sklearn.tree import DecisionTreeClassifier
from Using_CNN import feat_trainCNN,feat_testCNN,y_train,y_test
import numpy as np

clf = DecisionTreeClassifier(random_state=0)
clf = clf.fit(feat_trainCNN,np.argmax(y_train,axis=1))

TrainDecisionScoreCNN=clf.score(feat_trainCNN,np.argmax(y_train,axis=1))*100
print("Decision Tree Training Accuracy Score:-",TrainDecisionScoreCNN)


TestDecisionScoreCNN=clf.score(feat_testCNN,np.argmax(y_test,axis=1))*100
print("\nDecision Tree Testing Accuracy Score:-",TestDecisionScoreCNN)
