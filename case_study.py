# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 13:38:08 2020

@author: user
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.model_selection  import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics


dataset = pd.read_csv('sample.csv' , encoding='latin-1')


import seaborn as sns
corr = dataset.corr()
plt.figure(figsize=(16, 14))
sns.heatmap(corr, vmax=0.5, center=0,
            square=True, linewidths=2, cmap='Blues')
plt.savefig("heatmap.png")
plt.show()

dataset.corr()['loan_status']


print(f"{dataset.dtypes}\n")
print(f"Sum of null values in each feature:\n{35 * '-'}")
print(f"{dataset.isnull().sum()}")
dataset.head()


# Get number of positve and negative examples
#pos = dataset[loan_status.Fully Paid] == 1].shape[0]
#neg = dataset[dataset["loan_status"] == 0].shape[0]
#print(f"Positive examples = {pos}")
#print(f"Negative examples = {neg}")
#print(f"Proportion of positive to negative examples = {(pos / neg) * 100:.2f}%")



dataset.describe(include=[np.object])
col_names=list(dataset.columns)
for i in col_names:
    j=dataset[i].value_counts()
    print('-----------------------------------')
    print(j)


for m in col_names:
    dataset[m].hist()
    plt.show()
    
    
dataset.describe(include=[np.number])
dataset.info()    
dataset.isnull().sum()


col_names[14:15]
