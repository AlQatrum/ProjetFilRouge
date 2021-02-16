# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 09:13:44 2021

@author: alexi
"""

##### Using Adaboost #####

### Libraries ###1
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import itertools as it
from sklearn.metrics import confusion_matrix as CM
#from sklearn.datasets import make_classification

### Getting data ### 
Rep = 'C:/Users/alexi/OneDrive/Documents/DocumentsPersonnels/SynchroDropbox/Dropbox/IODAA2020/Agro/ProjetFilRouge/'
Data = pd.read_csv(Rep+'RemodelingData/GenusWithAgeClassesFull.csv')

### Changing age classes
Ages = Data['Age'].tolist()
AgeClasses = []
for i in range(len(Ages)):
    Age = Ages[i]
    if Age <= 10:
        AgeClasses.append('0;10')
    elif Age <= 20:
        AgeClasses.append('10;20')
    elif Age <= 60:
        AgeClasses.append('20;60')
    elif Age <= 80:
        AgeClasses.append('60;80')
    else:
        AgeClasses.append('80;104')

Data['AgeClass'] = AgeClasses

### Removing intermediate classes
Data = Data[Data.AgeClass != '10;20']
Data = Data[Data.AgeClass != '60;80']

### Balancing the classes
SmallestNbIndiv = len(Data[Data.AgeClass == '0;10'].index)
TrainIndivToSample = int(2*SmallestNbIndiv/3)

XTest = Data
XTrain = Data[Data.AgeClass == '0;10'].sample(n = TrainIndivToSample, axis = 0)
XTrain = XTrain.append(Data[Data.AgeClass == '20;60'].sample(n = TrainIndivToSample, axis = 0))
XTrain = XTrain.append(Data[Data.AgeClass == '80;104'].sample(n = TrainIndivToSample, axis = 0))
XTest = XTest.drop(XTrain.index)
XTrain = XTrain.sample(frac=1)
XTest = XTest.sample(frac=1)

YTrain = list(it.chain.from_iterable(XTrain[['AgeClass']].values.tolist()))
YTest = list(it.chain.from_iterable(XTest[['AgeClass']].values.tolist()))

XTrain = XTrain.drop(['Subject', 'Age', 'Gender', 'AgeClass'], 1)
XTest = XTest.drop(['Subject', 'Age', 'Gender', 'AgeClass'], 1)

#BaseEstimator = DecisionTreeClassifier(max_depth=None)
BaseEstimator = RandomForestClassifier(n_estimators=100, criterion='entropy', min_samples_leaf=5)

AdB = AdaBoostClassifier(base_estimator = BaseEstimator, n_estimators=500,algorithm="SAMME.R", learning_rate=0.5)
AdB.fit(XTrain, YTrain)

YPred = AdB.predict(XTest)

print("Performance:",sum(YPred==YTest)/len(YTest))
print("Confusion Matrix:\n",CM(YTest,YPred))

plot_confusion_matrix(AdB, XTest, YTest, cmap=plt.cm.Blues, normalize='true')

