#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 10:24:51 2021

@author: roulletagathe
"""

import pandas as pds
import numpy as np
import sklearn 
print(sklearn.__version__)
from sklearn.linear_model import LassoCV

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import scipy.stats as stat

#preparation des données

file = pds.read_csv("genus_level2.csv", index_col = 0, sep =",")
df  = pds.read_excel ("age_and_gender_jap.xlsx", index_col = 0)
df_inverse = np.transpose(df)
combined_csv = pds.concat([file, df_inverse])
df = combined_csv
df = df.drop(["gender", "BioSample No.", "number of high quality \nreads for 16S microbiota analysis " ])
bd_final = df.drop(["age"])
bd_final = np.transpose(bd_final)
X = bd_final
Y = df.iloc[-1,:]
feature_names = list(bd_final)

#initialisation des listes contenant les scores
score_RF = [] 
score_RN = []
score_lasso = []

#On fait tourner les algorithmes 10 fois pour avoir des résultats plus pertinents. 
for i in range(0,10):
    #construction du jeu de données de test et d'entrainement
    X_train, X_test, y_train, y_test = train_test_split(X, Y.astype(str), test_size = 0.3)
    
    #Lasso
    clf = LassoCV(cv = 10).fit(X_train, y_train)
    score = clf.score(X_test,y_test)
    score_lasso = score_lasso + [score]
    
    #RandomForest
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    predictions = clf.predict(X_test)
    
    #a decommenter pour avoir les matrices de confusion
    #print(confusion_matrix(y_test, predictions))
    #print(score)
    #print(classification_report(y_test,predictions))
    score_RF = score_RF + [score]
    
    #reseau de neurones
    clf = MLPClassifier(hidden_layer_sizes=(50,30,30,30,50), max_iter=1000).fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    predictions = clf.predict(X_test)

    #a decommenter pour avoir les matrices de confusion
    #print(confusion_matrix(y_test, predictions))
    #print(score)
    #print(classification_report(y_test,predictions))
    score_RN = score_RN + [score]


#lasso
print(" ")
print("Résultat du lasso sur les données sans classe d'age")
clf = LassoCV(cv = 10).fit(X_train, y_train)
importance = np.abs(clf.coef_)
idx_features = (-importance).argsort()[:5]
name_features = np.array(feature_names)[idx_features]
print('Selected features Lasso : {}'.format(name_features))

#RF
print(" ")
print("Résultat du random forest sur les données sans classe d'age")
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
importance = clf.feature_importances_
idx_features = (-importance).argsort()[:5]
name_features = np.array(feature_names)[idx_features]
print('Selected features Random Forest : {}'.format(name_features))


#On plot nos résultats 
boxplotElements = plt.boxplot([score_RF,
                                  score_RN], sym = 'g*')
plt.gca().xaxis.set_ticklabels(['Precision RF', 'Precision RN'])
plt.title(' Résultats sur 10 répétitions. Entrainement sur 70% des données.')
plt.suptitle('Précision des random forests (RF) et des réseaux de neurones (RN) pour des âges continus (chaque âge est une classe).')
for element in boxplotElements['medians']:
    element.set_color('blue')
    element.set_linewidth(4)
for element in boxplotElements['boxes']:
    element.set_color('red')
    element.set_linewidth(3)
for element in boxplotElements['whiskers']:
    element.set_color('red')
    element.set_linewidth(2)
for element in boxplotElements['caps']:
    element.set_color('red')
     
print(stat.ttest_ind(score_RF,score_RN)) # test de student, abscence de différence significative