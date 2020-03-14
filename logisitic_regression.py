# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:15:41 2020

@author: user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import the data
dataset = pd.read_csv('social_network.csv')

x = dataset.iloc[:,[2,3]].values#here X is Independent varriable 
y = dataset.iloc[:,4].values#here Y is dependent varriable

#spliting the dataset into train and test 
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 0)

#feature scalling to transfer the value -2 to 2 

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#create the model of logsitic Regression 

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train,y_train)

#Predict the test set result 
y_pred = classifier.predict(x_test)


#making confusing metrics to check predict value how much is correct 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)



