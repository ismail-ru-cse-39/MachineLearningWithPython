# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 17:35:27 2021

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

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/cell_samples.csv'
#wget.download(url, '/FuelConsumption.csv')

response = urlopen(url)
#cr = csv.reader(response)
cr = pd.read_csv(response, delimiter=',')

print(cr.head())
print(cr.shape)

ax = cr[cr['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cr[cr['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.show()

print(cr.dtypes)

cr = cr[pd.to_numeric(cr['BareNuc'], errors='coerce').notnull()]
cr = cr.astype({"BareNuc": int})

print(cr.dtypes)