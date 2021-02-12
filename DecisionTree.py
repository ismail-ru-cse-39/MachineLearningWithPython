# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 00:50:00 2021

@author: Ismail
"""
import numpy as np
import pandas as pd
from urllib.request import urlopen
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from  io import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree


url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
#wget.download(url, '/FuelConsumption.csv')

response = urlopen(url)
#cr = csv.reader(response)
cr = pd.read_csv(response, delimiter=',')

print(cr.head())
print(cr.shape)

X = cr[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
print(X[0:5])

#preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

print(X[0:5])

y = cr['Drug']
print(y[0:5])

#Setting up the decision tree

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

print(X_trainset.shape)
print(y_trainset.shape)
print("Shape of y: ")
print(X_testset.shape)
print(y_testset.shape)


#Modeling
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
print(drugTree) # it shows the default parameters

drugTree.fit(X_trainset,y_trainset)

predTree = drugTree.predict(X_testset)

print ("Predict Tree: ", predTree [0:5])
print ("y_test Tree: ", y_testset [0:5])

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

#visualization
dot_data = StringIO()
filename = "drugtree.png"
featureNames = cr.columns[0:5]
targetNames = cr["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')

