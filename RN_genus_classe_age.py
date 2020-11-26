# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 09:28:51 2020

@author: julie
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import pandas as pd


data=pd.read_csv("genus_classe_age.csv", sep=",")

X=data.drop("classe_age", axis=1)
Y=data["classe_age"]


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(120,120,120,120))

mlp.fit(X_train,Y_train)
predictions = mlp.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(Y_test, predictions))

print(classification_report(Y_test,predictions))