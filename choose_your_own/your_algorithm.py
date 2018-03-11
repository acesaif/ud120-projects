#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()

################################################################################

### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
"""
def NaiiveBayesVisualization():
	from sklearn.naive_bayes import GaussianNB
	from sklearn.metrics import accuracy_score
	import numpy as np

	clf = GaussianNB()
	clf.fit(features_train,labels_train)

	pred = clf.predict(features_test)
	# print(pred)

	accuracy = accuracy_score(pred,labels_test)
	print(accuracy)

NaiiveBayesVisualization()

def SVM_Visualization():
	from sklearn.svm import SVC
	from sklearn.metrics import accuracy_score
	import numpy as np

	clf = SVC(C=5000,kernel="rbf",gamma=2)
	clf.fit(features_train,labels_train)

	pred = clf.predict(features_test)
	# print(pred)

	accuracy = accuracy_score(pred,labels_test)
	print(accuracy)

SVM_Visualization()

def decisiontree_vis():
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.metrics import accuracy_score
	import numpy as np

	clf1 = DecisionTreeClassifier(min_samples_split=2)
	clf1.fit(features_train,labels_train)
	pred1 = clf1.predict(features_test)
	# print(pred1)
	acc_min_samples_split_2 = round(accuracy_score(pred1,labels_test),3)

	clf2 = DecisionTreeClassifier(min_samples_split=50)
	clf2.fit(features_train,labels_train)
	pred2 = clf2.predict(features_test)
	# print(pred2)
	acc_min_samples_split_50 = round(accuracy_score(pred2,labels_test),3)

	def acc():
		acc_2_50 = [acc_min_samples_split_2,acc_min_samples_split_50]
		return acc_2_50
	return acc()

print(decisiontree_vis())
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(features_train,labels_train)

pred = clf.predict(features_test)
print(pred)

acc = accuracy_score(pred,labels_test)
print(acc)

try:
	prettyPicture(clf, features_test, labels_test)
except NameError:
	pass
