# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 14:56:42 2021

@author: Hello
"""

#bibliothèque utiles
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importation des données
df=pd.read_csv('C:\\Users\\Hello\\Desktop\\IODAA\\fil_rouge\\ratio_genre_trié_avec_age_transpose.csv', sep=',', header=0 )
X=df.drop(columns=['age',"Unnamed: 0"])
Y=df["age"]


#fabrication des ensemble de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.5)


from sklearn.linear_model import LassoCV


#lasso
print(" ")
print("Résultat du lasso sur les ratios de genre sans classe d'age")
clf = LassoCV(cv = 100).fit(X_train, y_train)
importance = np.abs(clf.coef_)


idx_third = importance.argsort()[-3]
threshold = importance[idx_third] + 0.01

#attribut sélectionnés
idx_features = (-importance).argsort()[:20]
feature_names = list(df)
name_features = np.array(feature_names)[idx_features]
print('Selected features: {}'.format(name_features))

#score du lasso
mean_square = clf.score(X_test,y_test)
print("Score Lasso ensemble test: ", mean_square)