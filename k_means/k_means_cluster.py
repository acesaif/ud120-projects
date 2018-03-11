#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()

### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)


### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2, feature_3]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )


### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
for f1, f2, f3 in finance_features:
    plt.scatter( f1, f2 )
plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

kmeans = KMeans(n_clusters=2)
pred = kmeans.fit_predict(finance_features)

### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"

exer_stock = []
for i in data_dict:
    exer_stock.append(data_dict[i]['exercised_stock_options'])

stock_no_Nan = [x for x in exer_stock if x != 'NaN']
print max(stock_no_Nan)
print min(stock_no_Nan)
print

sal = []
for i in data_dict:
    sal.append(data_dict[i]['salary'])

sal_no_Nan = [x for x in sal if x != 'NaN']
print max(sal_no_Nan)
print min(sal_no_Nan)
print
"""
salarr = np.array(sal_no_Nan,dtype=np.float64)
salscaler = MinMaxScaler()
rescale_sal = salscaler.fit_transform(salarr)
print rescale_sal
"""
def featureScaling(arr):
    try:
        rescaled = []
        for i in arr:
            rescaled.append((i-min(arr))/(float(max(arr)-min(arr))))
        return rescaled
    except ZeroDivisionError:
        return 'Division by zero mathematically not possible.'

print featureScaling(sal_no_Nan)
print

from sklearn.preprocessing import MinMaxScaler

weights = np.array([[115,15],[140,14],[175,13]],dtype=np.float64)
scaler = MinMaxScaler()
rescaled = scaler.fit_transform(weights)
print rescaled
print


