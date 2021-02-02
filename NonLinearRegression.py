# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 20:49:00 2021

@author: Ismail
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from urllib.request import urlopen
import csv
from scipy.optimize import curve_fit

x = np.arange(-5.0, 5.0, 0.1)
y = 2 * (x) + 3
y_noise = 2 * np.random.normal(size = x.size)
ydata = y + y_noise

# plt.plot(x, ydata, 'bo')
# plt.plot(x, y, '-r')
# plt.ylabel('Dependent variable')
# plt.xlabel('Independent variable')
# plt.show()

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/china_gdp.csv'
#wget.download(url, '/FuelConsumption.csv')

response = urlopen(url)
#cr = csv.reader(response)
cr = pd.read_csv(response)

print(cr.head(10))

print(cr.describe())

plt.figure(figsize=(8,5))

x_data, y_data = (cr['Year'].values, cr["Value"].values)
plt.plot(x_data, y_data, 'ro')
plt.xlabel('GDP')
plt.ylabel('Year')
plt.show()

#Model
X = np.arange(-5.0, 5.0, 0.1)
Y = 1.0/ (1.0 + np.exp(-X))

plt.plot(X, Y)
plt.ylabel('Dependent variable')
plt.xlabel('Independent variable')
plt.show()

def sigmoid(x, beta1, beta2):
    y = 1 / (1 + np.exp(-beta1 * (x - beta2)))
    return y

beta1 = 0.10
beta2 = 1990.0

#logistic function
y_pred = sigmoid(x_data, beta1, beta2)

#plot initial prediction against datapoint
plt.plot(x_data, y_pred*15000000000000.)
plt.plot(x_data, y_data, 'ro')

#normalization
xdata = x_data/max(x_data)
ydata = y_data/max(y_data)

popt, pcov = curve_fit(sigmoid, xdata, ydata)
print("beta1 = %f, beta2 = %f" % (popt[0], popt[1]))

x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()






