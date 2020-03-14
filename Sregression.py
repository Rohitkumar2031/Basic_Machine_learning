# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:02:51 2020

@author: user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("salary_data.csv")

#devide the dataset in dependent and Independent varribales

x = dataset.iloc[:,:-1].values#[: use for row value and :-1 for column -1 drop last column]
y = dataset.iloc[:,1].values

#spliting the dataset into traning and testing dataset 

#from sklearn.cross_validation import train_test_split#this is a package to divide the train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)

#Impletemnet of classifier based on simple Linear Regression 

from sklearn.linear_model import LinearRegression#this package calculate B0 and B1 
simplelinearregression = LinearRegression()
simplelinearregression.fit(x_train,y_train)#here we set the data set which we want to fit the model 

y_predict1 = simplelinearregression.predict(x_test)
from sklearn.metrics import mean_squared_error
#print(mean_squared_error(x_train,y_train))
y_predict1 = simplelinearregression.predict([[11]])

# compare the predicted vaule to the orignal value
# The mean squared error
print("Mean squared error: %.2f" % np.mean((simplelinearregression.predict(x_test) - y_test) ** 2))

#print Varience score means error score 1 is perfect prrdecition
print('Variance score: %.2f' % simplelinearregression.score(x_test, y_test))


#y_predict_val1 = simplelinearregression.predict()

#Now plotiing the graph 

plt.scatter(x_train ,y_train,color = 'red')
plt.plot(x_train,simplelinearregression.predict(x_train))
plt.show()


plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,simplelinearregression.predict(x_train))
plt.title('Salary Vs Experince')
plt.xlabel('Experince')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_train,y_train, color = 'blue')
plt.plot(x_train,simplelinearregression.predict(x_train))
plt.show()




