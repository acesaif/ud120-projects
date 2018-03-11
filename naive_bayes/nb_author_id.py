#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
import numpy as np
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

print features_train.shape
print np.array(labels_train).shape
print features_test.shape
print np.array(labels_test).shape

#########################################################
### your code goes here ###

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np

clf = GaussianNB()
clf.fit(features_train,labels_train)

pred = clf.predict(features_test)
print(pred)

accuracy = accuracy_score(pred,labels_test)
print(accuracy)

t0 = time() # measuring time for training the algorithm.
clf.fit(features_train,labels_train)
print "Training time: %ss"%round(time()-t0,3)

t1 = time() # measuring time for predicting the algorithm.
clf.predict(features_test)
print "Prediction time %ss:"%round(time()-t1,3)

#########################################################


