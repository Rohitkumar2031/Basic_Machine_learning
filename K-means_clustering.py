# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 10:53:15 2020

@author: user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('mall_customer.csv')

x = dataset.iloc[:,[3,4]].values

#using the Elbow method to find the optimal number of cluster

from sklearn.cluster import KMeans

#wcss within cluster sum of square
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)#here inertia is itrate each range value and sum
    
plt.plot(range(1,11),wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of cluster")
plt.ylabel("WCSS")
plt.show()    

#to fit kmeans dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
#Y-kmeans mean all the cluster whose data required in cluster and distubuted bases on X 
y_kmeans = kmeans.fit_predict(x)


    