#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(C=10000,kernel="rbf")

# features_train = features_train[:len(features_train)/100] 
# labels_train = labels_train[:len(labels_train)/100]

clf.fit(features_train,labels_train)

pred = clf.predict(features_test)
print(pred)
# print(pred[10],pred[26],pred[50])

"""ones = []
for i in pred:
	if i == 1:
		ones.append(i)
print("Class 1: %s"%len(ones))"""

accuracy = accuracy_score(pred,labels_test)
print(accuracy)

to = time()
clf.fit(features_train,labels_train)
print "Training time: %ss"%round(time()-to,3)

t1 = time()
clf.predict(features_test)
print "Predicting time: %ss"%round(time()-t1,3)

#########################################################


