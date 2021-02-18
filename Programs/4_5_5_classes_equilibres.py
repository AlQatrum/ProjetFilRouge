#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 14:55:20 2021

@author: roulletagathe
"""
import pandas as pds
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import scipy.stats as stat
from sklearn.preprocessing import KBinsDiscretizer

#discretisation par 25

#Preparation des données
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

#répétition 10 fois pour chaque nombre de groupe
liste_nb_groupes = []
liste_moyenne_RN = []
liste_moyenne_RF = []
for i in range(2,23): 
    liste_nb_groupes = liste_nb_groupes + [i]
    enc = KBinsDiscretizer(n_bins=i, encode='ordinal', strategy = 'quantile')
    Y_array = np.asarray(Y).reshape(-1, 1)
    Y_trans = enc.fit_transform(Y_array)
    enc.bin_edges_[0]
    accuracy = []
    score_RF = []
    compteur = 0
    for i in range(0,10):
        
        X_train, X_test, y_train, y_test = train_test_split(X, Y_trans, test_size = 0.3)
        clfe = MLPClassifier(hidden_layer_sizes=(50,30,30,30,50), max_iter=1000).fit(X_train, y_train)
        #print(clfe)
        score = clfe.score(X_test, y_test)
        predictions = clfe.predict(X_test)
        #a decommenter pour avoir les matrices de confusions
        #print(confusion_matrix(y_test, predictions))
        #print(score)
        #print(classification_report(y_test,predictions))
        
        accuracy = accuracy + [score]
        compteur = compteur + 1
    #print(accuracy/compteur)

        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        #print("score sur ensemble de test RF avec classe",clf.score(X_test, y_test))
        # a decommenter pour avoir les features importantes
#        importance = clf.feature_importances_
#        idx_third = importance.argsort()[-3]
#        threshold = importance[idx_third] + 0.01
#        
#        idx_features = (-importance).argsort()[:5]
#        name_features = np.array(feature_names)[idx_features]
        #print('Selected features Random Forest : {}'.format(name_features))
        score = clf.score(X_test, y_test)
        predictions1 = clf.predict(X_test)
        #a decommenter pour avoir la matrice de confusion
        #print(confusion_matrix(y_test, predictions1))
        #print(score1)
        #print(classification_report(y_test,predictions1))
        score_RF = score_RF + [score]
    
    liste_moyenne_RN = liste_moyenne_RN + [accuracy]
    liste_moyenne_RF = liste_moyenne_RF + [score_RF]



boxplotElements = plt.boxplot(liste_moyenne_RN , sym = 'g*')
plt.gca().xaxis.set_ticklabels(liste_nb_groupes)
plt.title(' Résultats sur 10 répétitions. Entrainement sur 70% des données.')
plt.suptitle("Précision en fonction du nombre de groupe sur un RN. Les classes sont équilibrées")
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
plt.show()

boxplotElements = plt.boxplot(liste_moyenne_RF , sym = 'g*')
plt.gca().xaxis.set_ticklabels(liste_nb_groupes)
plt.title(' Résultats sur 10 répétitions. Entrainement sur 70% des données.')
plt.suptitle("Précision en fonction du nombre de groupe sur un RF. Les classes sont équilibrées")
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
plt.show()



#on recupère les features importantes 
enc = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy = 'quantile')
Y_array = np.asarray(Y).reshape(-1, 1)
Y_trans = enc.fit_transform(Y_array)
enc.bin_edges_[0]
X_train, X_test, y_train, y_test = train_test_split(X, Y_trans, test_size = 0.3)
print("Résultat du random forest sur les données avec classe d'âge")
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print("score sur ensemble de test RF avec classe",clf.score(X_test, y_test))

importance = clf.feature_importances_
idx_third = importance.argsort()[-3]
threshold = importance[idx_third] + 0.01
#        
idx_features = (-importance).argsort()[:5]
name_features = np.array(feature_names)[idx_features]
print('Selected features Random Forest : {}'.format(name_features))
score = clf.score(X_test, y_test)
predictions1 = clf.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions1))
print(score)
print(classification_report(y_test,predictions1))
print(enc.bin_edges_[0])