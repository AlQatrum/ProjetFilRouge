# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 09:12:48 2021

@author: julie
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import KBinsDiscretizer
from statistics import *


#ipmportation des données des genres
file = pds.read_csv("genus_level2.csv", index_col = 0, sep =",")

#importation de la table des individus
df  = pds.read_excel ("age_and_gender_jap.xlsx", index_col = 0)

#modification des tables (transposition pour fusion)
df_inverse = np.transpose(df)
combined_csv = pds.concat([file, df_inverse])
df = combined_csv
#suppression des colonnes sans intérêt
df = df.drop(["gender", "BioSample No.", "number of high quality \nreads for 16S microbiota analysis " ])
df = np.transpose(df)

#initialisation de liste vide où stocker les scores de chaque itération
score_RF=[]

#création de 2 bases pour les trois classes
#bébé : 0 à 1,5 ans
#sénior : 85 ans et plus
base_1=df[df["age"]<1.5]
base_2=df[df["age"]>=85]
#concaténation des deux bases
bd_final=pds.concat([base_1, base_2])


#suppression de l'âge pour la table des variables explicatives du modèle
X=bd_final.drop(["age"], axis=1)

#ne conserver que l'âge pour la table des étiquettes
bd_Y = np.transpose(bd_final)
Y=bd_Y.iloc[-1,:]

#parcours de la table des étiquettes pour ajouter un nom à la classe
for i in range(0,len(Y)):
    if Y[i]<1.5 : 
        Y.iloc[i]= str("bebe")
    else :
        Y.iloc[i]= str("sénior")

#boucle for pour faire 30 itérations
for i in range(0,30):      
    
    #séparation des données en ensemble d'apprentissage et de test (70%-30%)
    X_train, X_test, y_train, y_test = train_test_split(X, Y.astype(str), test_size = 0.3)
    
    feature_names = list(bd_final)
    print(feature_names)


    ##random forest avec nombre d'arbre par défaut
    from sklearn.datasets import make_classification
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    #affichage du score
    print("score sur ensemble de test RF avec classe",clf.score(X_test, y_test))
    
    #selection des variables qui jouent un rôle plus important dans la construction du modèle
    importance = clf.feature_importances_
    idx_third = importance.argsort()[-3]
    threshold = importance[idx_third] + 0.01
    
    idx_features = (-importance).argsort()[:5]
    name_features = np.array(feature_names)[idx_features]
    print('Selected features Random Forest : {}'.format(name_features))
    
    #affichage de la matrice de confusion, du score de précision et de la table des classifications
    score1 = clf.score(X_test, y_test)
    predictions1 = clf.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test, predictions1))
    print(score1)
    print(classification_report(y_test,predictions1))
    
    #ajout du score à la liste des scores
    score_RF.append(score1)
    
    #plot de matrice de confusion
    plt.rcParams.update({'font.size': 12})
    sklearn.metrics.plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Blues)
  
    
#boxplot des précisions
boxplotElements = plt.boxplot([score_RF], sym = 'g*')
plt.title(' Résultats sur 30 répétitions. Entrainement sur 70% des données.')
plt.suptitle("Précision d'un RF sur deux classes : 0-1,5 et >85 ans.")
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