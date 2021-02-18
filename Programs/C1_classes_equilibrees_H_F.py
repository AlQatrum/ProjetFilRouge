#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 15:02:58 2021

@author: roulletagathe
"""
import pandas as pds
import numpy as np

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import KBinsDiscretizer
import scipy.stats as stat

#resultats pour les femmes
file = pds.read_csv("genus_level2.csv", index_col = 0, sep =",")
df  = pds.read_excel ("age_and_gender_jap.xlsx", index_col = 0)
df_inverse = np.transpose(df)
combined_csv = pds.concat([file, df_inverse])
df = combined_csv
df = np.transpose(df)
df = df.loc[df["gender"] == 'female']
df = np.transpose(df)


df = df.drop(["gender", "BioSample No.", "number of high quality \nreads for 16S microbiota analysis " ])

bd_final = df.drop(["age"])
bd_final = np.transpose(bd_final)
X = bd_final
Y = df.iloc[-1,:]

liste_nb_groupes = []
liste_moyenne = []


liste_nb_groupes = []
liste_moyenne_homme = []
liste_moyenne_femme = []

enc = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy = 'quantile')
Y_array = np.asarray(Y).reshape(-1, 1)
Y_trans = enc.fit_transform(Y_array)
classe_femme = enc.bin_edges_[0]
accuracy = []
score_RF = []
compteur = 0
for i in range(0,10):
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y_trans, test_size = 0.3)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    #a decommenter pour avoir les features interessants
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
# a decommenter pour avoir les matrices de confusions
    #print(confusion_matrix(y_test, predictions1))
    #print(score1)
    #print(classification_report(y_test,predictions1))
    liste_moyenne_femme = liste_moyenne_femme + [score]




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


enc = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy = 'quantile')
Y_array = np.asarray(Y).reshape(-1, 1)
Y_trans = enc.fit_transform(Y_array)
classe_homme = enc.bin_edges_[0]
accuracy = []
score_RF = []
compteur = 0
for i in range(0,10):
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y_trans, test_size = 0.3)
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
# à decommenter pour avoir la matrice de confusion
    #print(confusion_matrix(y_test, predictions1))
    #print(score1)
    #print(classification_report(y_test,predictions1))
    liste_moyenne_homme = liste_moyenne_homme + [score]



boxplotElements = plt.boxplot([liste_moyenne_femme, liste_moyenne_homme], sym = 'g*')
plt.gca().xaxis.set_ticklabels(["femme","homme"])
plt.title(' Résultats sur 10 répétitions. Entrainement sur 70% des données.')
plt.suptitle("Précision en fonction du sexe sur un RF. Les classes sont équilibrées")
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

print(classe_femme, classe_homme) #age de separations pour respectivement les femmes et les hommes

#test statistique
print(stat.ttest_ind(liste_moyenne_femme,liste_moyenne_homme)) 