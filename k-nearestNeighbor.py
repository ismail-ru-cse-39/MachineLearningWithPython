# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:52:07 2021

@author: Ismail
"""
#k-Nearest neighbor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from urllib.request import urlopen
import csv
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv'
#wget.download(url, '/FuelConsumption.csv')

response = urlopen(url)
#cr = csv.reader(response)
cr = pd.read_csv(response)

print(cr.head())

print(cr['custcat'].value_counts())

cr.hist(column = 'income', bins = 50)

print(cr.columns)
X = cr[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
#print(X[0:5])

y = cr['custcat'].values
print(y[0:5])

#Normalize data

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
print(X[0:5])

#Train Test data split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

#KNN
k = 4
#train model and predict
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
print(neigh)

yhat = neigh.predict(X_test)

print(yhat[0:5])


#Accuracy Evaluation
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))