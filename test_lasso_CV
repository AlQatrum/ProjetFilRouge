#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:32:34 2020

@author: roulletagathe
"""

import pandas as pds
import numpy as np
import sklearn 
print(sklearn.__version__)
from sklearn.linear_model import LassoCV

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score

file = pds.read_csv("genus_level2.csv", index_col = 0, sep =",")
df  = pds.read_excel ("age_and_gender_jap.xlsx", index_col = 0)
df_inverse = np.transpose(df)
combined_csv = pds.concat([file,df_inverse])
df = combined_csv

df = df.drop(["gender", "BioSample No.", "number of high quality \nreads for 16S microbiota analysis " ])

bd_final = df.drop(["age"])
bd_final = np.transpose(bd_final)

X = bd_final
Y = df.iloc[-1,:]
Y=Y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.5)

feature_names = list(bd_final)

#lasso
print(" ")
print("Résultat du lasso sur les données sans classe d'age")
clf = LassoCV(cv = 100).fit(X_train, y_train)
importance = np.abs(clf.coef_)


idx_third = importance.argsort()[-3]
threshold = importance[idx_third] + 0.01

idx_features = (-importance).argsort()[:20]
name_features = np.array(feature_names)[idx_features]
print('Selected features: {}'.format(name_features))

mean_square = clf.score(X_test,y_test)
print("Score Lasso ensemble test: ", mean_square)


##random forest
from sklearn.datasets import make_classification
print(" ")
print("Résultat du random forest sur les données sans classe d'âge")
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print("score sur ensemble de test RF sans classe",clf.score(X_test, y_test))



importance = clf.feature_importances_
idx_third = importance.argsort()[-3]
threshold = importance[idx_third] + 0.01

idx_features = (-importance).argsort()[:20]
name_features = np.array(feature_names)[idx_features]
print('Selected features Random Forest : {}'.format(name_features))


##random forest with class : 

file_classe = pds.read_csv("table_age_2_sex_genre.csv", index_col = 0, sep =",")
X = file_classe[file_classe.columns.difference(["Subject", "age", "gender"])]
y = file_classe["age"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

clf = RandomForestClassifier()
clf.fit(X, y)
#print("score sur ensemble de test",clf.score(X_test, y_test))
print(" ")
print("Résultat du random forest sur les données avec classe d'âge (par décade)")
scores = cross_val_score(clf, X, y, cv=5)
print("score CV test par classe d'age", scores.mean())


importance = clf.feature_importances_
idx_third = importance.argsort()[-3]
threshold = importance[idx_third] + 0.01

idx_features = (-importance).argsort()[:20]
name_features = np.array(feature_names)[idx_features]
print('Selected features Random Forest : {}'.format(name_features))

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print("score sur ensemble de test RF avec classe",clf.score(X_test, y_test))
