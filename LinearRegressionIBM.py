# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 17:09:50 2021

@author: Ismail
"""
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import wget
from urllib.request import urlopen
import csv
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

#%matplotlib inline

#!wget -O FuelConsumption.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv


url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv'
#wget.download(url, '/FuelConsumption.csv')

response = urlopen(url)
#cr = csv.reader(response)
cr = pd.read_csv(response)

print(cr.head())

print(cr.describe())

cdf = cr[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(cdf.head(9))

viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
#viz.hist()
#plt.show()

# plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
# plt.xlabel("FUELCONSUMPTION_COMB")
# plt.ylabel("EMISSION")
# plt.show()

# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='red')
# plt.xlabel("Engine size")
# plt.ylabel('Emission')
# plt.show()

# plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='green')
# plt.xlabel("Cylinder")
# plt.ylabel("Emissions")
# plt.show()

#Spliting train and test data set
msk = np.random.rand(len(cdf)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='yellow')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()


#modeling

regr = linear_model.LinearRegression()
train_x = np.asanyarray(train['ENGINESIZE'])
train_y = np.asanyarray(train['CO2EMISSIONS'])
#print(train_x)
train_x = train_x.reshape((-1, 1))
train_y = train_y.reshape((-1, 1))
regr.fit(train_x,train_y)
#print coefficient
print('Co-efficient', regr.coef_)
print('Intercept', regr.intercept_)

# plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
# plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
# plt.xlabel("Engine size")
# plt.ylabel('Emission')


#Evaluation the accuracy

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of square(MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score %.2f" % r2_score(test_y, test_y_))

poly3 = PolynomialFeatures(degree = 3)
train_x_poly3 = poly3.fit_transform(train_x)
clf3 = linear_model.LinearRegression()
train_y3_ = clf3.fit(train_x_poly3, train_y)

#The co-efficient
print('Coefficient: ', clf3.coef_)
print('Intercept: ', clf3.intercept_)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color = 'green')
XX = np.arange(0.0, 10.0, 0.1)
YY = clf3.intercept_[0] + clf3.coef_[0][1]*XX + clf3.coef_[0][2]*np.power(XX,2)
+ clf3.coef_[0][3] * np.power(XX, 3)
plt.plot(XX, YY, '-r')


