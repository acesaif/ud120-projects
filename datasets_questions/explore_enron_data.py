#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import pandas as pd

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
# print enron_data

total = []
for pay in enron_data:
	total.append(enron_data[pay]['total_payments'])
print total
print len(total)
print

totnan = [x for x in total if x != 'NaN']
print totnan
print max(totnan)
print

nan_tot = [p for p in total if p == 'NaN']
print len(nan_tot)
print

print (float(len(nan_tot))/len(total) * 100)
print

for name in enron_data:
	if enron_data[name]['total_payments'] == 309886585:
		print name
print

df = pd.DataFrame(enron_data)
# print df
print df.loc['poi'].count()
print

print len(enron_data) # size of dataset
print

print len(enron_data['METTS MARK']) # no. of features
print

count = 0
for i in enron_data:
	if enron_data[i]["poi"] == 1:
		count += 1
print count # count of poi's == True
print

print enron_data["PRENTICE JAMES"]["total_stock_value"]
print

def easier(string):
	space_splitter = string.split(" ")
	capitalized = []
	j = 0
	for i in space_splitter:
		capitalized.append(i.upper())
	if len(capitalized) == 2:
		temp = capitalized[1]
		capitalized[1] = capitalized[0]
		capitalized[0] = temp
	elif len(capitalized) == 3:
		pemp = capitalized[j+1]
		capitalized[j+1] = capitalized[j+2]
		capitalized[j+2] = pemp
		gemp = capitalized[j+1]
		capitalized[j+1] = capitalized[j]
		capitalized[j] = gemp
	return " ".join(capitalized)

print df.loc[:,easier("Wesley Colwell")]
print

print df.loc[:,easier("Jeffrey K Skilling")]
print

print df.loc[:,easier("Kenneth L Lay")]
print

print df.loc[:,easier("Andrew S Fastow")]
print

print enron_data[easier("Jeffrey K Skilling")]["total_payments"]
print enron_data[easier("Kenneth L Lay")]["total_payments"]
print enron_data[easier("Andrew S Fastow")]["total_payments"]
print

salary_list = []
for s in enron_data:
	if enron_data[s]['salary'] != 'NaN':
		salary_list.append(enron_data[s]['salary'])
print salary_list
print len(salary_list)
print

email = []
for mail in enron_data:
	if enron_data[mail]['email_address'] != 'NaN':
		email.append(enron_data[mail]['email_address'])
print email
print len(email)
print

count = 0
for pois in enron_data:
	if enron_data[pois]['poi'] == 1:
		if enron_data[pois]['total_payments'] == 'NaN':
			count += 1
print count
print

number = 0
for tp in enron_data:
	if enron_data[tp]['total_payments'] == "NaN":
		number += 1
print number
print

enron_data.pop('TOTAL',None	)

exer_stock = []
for i in enron_data:
	exer_stock.append(enron_data[i]["exercised_stock_options"])

exer_stoc_noNan = [na for na in exer_stock if na != "NaN"]
print exer_stoc_noNan
print
print max(exer_stoc_noNan)
print min(exer_stoc_noNan)
print

