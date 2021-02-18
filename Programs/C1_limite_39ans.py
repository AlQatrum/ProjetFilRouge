#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 10:25:46 2021

@author: roulletagathe
"""

import pandas as pds
import numpy as np


from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import classification_report, confusion_matrix

#resultats pour les femmes
file = pds.read_csv("genus_level2.csv", index_col = 0, sep =",")
df  = pds.read_excel ("age_and_gender_jap.xlsx", index_col = 0)
df_inverse = np.transpose(df)
combined_csv = pds.concat([file, df_inverse])
df = combined_csv
df = np.transpose(df)
print(df.shape)
df = df.loc[df["gender"] == 'female']
print(df.shape)
df = np.transpose(df)
df = df.drop(["gender", "BioSample No.", "number of high quality \nreads for 16S microbiota analysis " ])
bd_final = df.drop(["age"])
bd_final = np.transpose(bd_final)
X = bd_final
Y = df.iloc[-1,:]

feature_names = list(bd_final)

for i in range(0,len(Y)):
    if Y[i]<39 : 
        Y.iloc[i]= str("0-39")
    else : 
        Y.iloc[i]= str("40-109")

liste_nb_groupes = []
liste_moyenne = []


liste_nb_groupes = []
liste_moyenne_homme = []
liste_moyenne_femme = []
accuracy = []
score_RF = []
compteur = 0
for i in range(0,10):
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
    print(" ")
    print("Résultat du random forest sur les données avec classe d'âge")
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    #a decommenter pour avoir les features importantes
    #print("score sur ensemble de test RF avec classe",clf.score(X_test, y_test))
    
#        importance = clf.feature_importances_
#        idx_third = importance.argsort()[-3]
#        threshold = importance[idx_third] + 0.01
#        
#        idx_features = (-importance).argsort()[:5]
#        name_features = np.array(feature_names)[idx_features]
    #print('Selected features Random Forest : {}'.format(name_features))
    score = clf.score(X_test, y_test)
    predictions1 = clf.predict(X_test)
#a decommenter pour avoir les matrices de confusions
    #print(confusion_matrix(y_test, predictions1))
    #print(score1)
    #print(classification_report(y_test,predictions1))
    liste_moyenne_femme = liste_moyenne_femme + [score]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
print(" ")
print("Résultat du random forest sur les données avec classe d'âge")
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print("score sur ensemble de test RF avec classe",clf.score(X_test, y_test))
importance = clf.feature_importances_
idx_third = importance.argsort()[-3]
threshold = importance[idx_third] + 0.01
idx_features = (-importance).argsort()[:5]
name_features = np.array(feature_names)[idx_features]
print('Selected features Random Forest : {}'.format(name_features))
score = clf.score(X_test, y_test)
predictions1 = clf.predict(X_test)
print(confusion_matrix(y_test, predictions1))
print(score)
print(classification_report(y_test,predictions1))



#idem pour les hommes
file = pds.read_csv("genus_level2.csv", index_col = 0, sep =",")
df  = pds.read_excel ("age_and_gender_jap.xlsx", index_col = 0)
df_inverse = np.transpose(df)
combined_csv = pds.concat([file, df_inverse])
df = combined_csv
df = np.transpose(df)
df = df.loc[df["gender"] == 'male']
df = np.transpose(df)
df = df.drop(["gender", "BioSample No.", "number of high quality \nreads for 16S microbiota analysis " ])
bd_final = df.drop(["age"])
bd_final = np.transpose(bd_final)
X = bd_final
Y = df.iloc[-1,:]

for i in range(0,len(Y)):
    if Y[i]<39 : 
        Y.iloc[i]= str("0-39")
    else : 
        Y.iloc[i]= str("40-109")


accuracy = []
score_RF = []
compteur = 0
for i in range(0,10):
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    #a decommenter pour les features importantes
    #print("score sur ensemble de test RF avec classe",clf.score(X_test, y_test))
    
#        importance = clf.feature_importances_
#        idx_third = importance.argsort()[-3]
#        threshold = importance[idx_third] + 0.01
#        
#        idx_features = (-importance).argsort()[:5]
#        name_features = np.array(feature_names)[idx_features]
    #print('Selected features Random Forest : {}'.format(name_features))
    score = clf.score(X_test, y_test)
    predictions1 = clf.predict(X_test)
#a decommenter pour avoir les matrices de confusion
    #print(confusion_matrix(y_test, predictions1))
    #print(score1)
    #print(classification_report(y_test,predictions1))
    liste_moyenne_homme = liste_moyenne_homme + [score]



boxplotElements = plt.boxplot([liste_moyenne_femme, liste_moyenne_homme], sym = 'g*')
plt.gca().xaxis.set_ticklabels(["femme","homme"])
plt.title(' Résultats sur 10 répétitions. Entrainement sur 70% des données.')
plt.suptitle("Précision en fonction du sexe sur un RF. 2 classes : 0-39 ans/40-109 ans")
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
    

#test statistique
print(stat.ttest_ind(liste_moyenne_femme,liste_moyenne_homme)) 