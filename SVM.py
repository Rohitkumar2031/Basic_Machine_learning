# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:20:27 2020

@author: user
"""
#Import packages
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


#Read the dataset
dataset = pd.read_csv('social_network.csv')

#make dependent and independent varriables here x is independent and y is depenedet varriables
x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values

#dataset divide into test and train 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 0)

#Here we are using sclar to transfer the value -2 to 2 scalar basically doing not ignore minimum value 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#create the model of SVM 
from sklearn.svm import SVC
classifier = SVC(kernel='linear',random_state = 0)
classifier.fit(x_train,y_train)
#
#classifier = SVC(kernel='rbf',degree = 5,random_state = 0)
#classifier.fit(x_train,y_train)

#now predict the values 
y_pred = classifier.predict(x_test)

#now check how many accuracy in my predict values 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
#


