# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 15:08:08 2021

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import jaccard_score
import itertools
from sklearn.metrics import log_loss

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv'
#wget.download(url, '/FuelConsumption.csv')

response = urlopen(url)
#cr = csv.reader(response)
cr = pd.read_csv(response, delimiter=',')

print(cr.head())
print(cr.shape)


churn_df = cr[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
 
churn_df = churn_df.astype({"churn": int})

print("Converted churn: ")

print(churn_df.head())

X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
print("X :", X[0:5])

y = np.asarray(churn_df['churn'])
print("Y: ", y [0:5])

#Normalize the data set

X = preprocessing.StandardScaler().fit(X).transform(X)
print("X after normalization: ", X[0:5])


#Train and Test
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


#Modeling
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
#print(LR)
yhat = LR.predict(X_test)
print(yhat)

#pridict the probability
yhat_prob = LR.predict_proba(X_test)
print(yhat_prob)

#Evaluation
print(jaccard_score(y_test, yhat,pos_label=0))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')

print(log_loss(y_test, yhat_prob))
