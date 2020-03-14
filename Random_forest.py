# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 12:11:23 2020

@author: user
"""
#step 1 import the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#2 step importing the dataset
dataset = pd.read_csv('position_salary.csv')
#step 3 divide the dataset in x and y 
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#split the dataset in train and test but we dont need to this data set bcz dataset is very small

#from sklearn.model_selection import train_test_split
#x_train,x_test,y_train,y_text = train_test_split(x,y,test_size = 0.3,random_state = 0)

#fitting Random Forest Regression to the dataset apply random forest
from sklearn.ensemble import RandomForestRegressor
rohit = RandomForestRegressor(n_estimators = 100,random_state = 0)
rohit.fit(x,y)

#Here we predict the value how many experince and ho many salary will received this is predict
y_pred = rohit.predict([[6.5]])

#now plot the graph 
plt.scatter(x,y,color = 'red')
plt.plot(x,rohit.predict(x), color = 'blue')
plt.show()

dataset.describe, 2
