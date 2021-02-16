# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 15:45:48 2021

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

#importation des données sur le genre
file = pds.read_csv("genus_level2.csv", index_col = 0, sep =",")

#importation des individus
df  = pds.read_excel ("age_and_gender_jap.xlsx", index_col = 0)

#modification des tables (transposition puis fusion des deux)
df_inverse = np.transpose(df)
combined_csv = pds.concat([file, df_inverse])
df = combined_csv
df = df.drop(["gender", "BioSample No.", "number of high quality \nreads for 16S microbiota analysis " ])
df = np.transpose(df)

#initialisation de liste de score pour les stocker au fil des itérations
score_RF=[]
score_RN=[]

#création d'une boucle for pour faire 10 apprentissage sur 10 sélections aléatoires de 25 individus
for tour in range(0,10):
    #création de 10 bases pour les 10 classes d'âge avec une sélection aléatoire de 25 individus
    #sauf pour la classe 2 qui n'a pas assez d'individus
    base_1=df[df["age"]<10]
    base_1=base_1.sample(25)
    base_2=df[(df["age"]>=10) & (df["age"]<20)]
    base_3=df[(df["age"]>=20) & (df["age"]<30)]
    base_3=base_3.sample(25)
    base_4=df[(df["age"]>=30) & (df["age"]<40)]
    base_4=base_4.sample(25)
    base_5=df[(df["age"]>=40) & (df["age"]<50)]
    base_5=base_5.sample(25)
    base_6=df[(df["age"]>=50) & (df["age"]<60)]
    base_6=base_6.sample(25)
    base_7=df[(df["age"]>=60) & (df["age"]<70)]
    base_7=base_7.sample(25)
    base_8=df[(df["age"]>=70) & (df["age"]<80)]
    base_8=base_8.sample(25)
    base_9=df[(df["age"]>=80) & (df["age"]<90)]
    base_9=base_9.sample(25)
    base_10=df[df["age"]>=90]
    base_10=base_10.sample(25)

    #concaténation des 10 bases dans une seule table
    bd_final=pds.concat([base_1, base_2, base_3, base_4, base_5, base_6, base_7, base_8, base_9, base_10])

    #suppression de l'âge dans la table des variables explicatives
    X=bd_final.drop(["age"], axis=1)
    
    #garder uniquement l'âge pour la table des étiquettes
    bd_Y = np.transpose(bd_final)
    Y=bd_Y.iloc[-1,:]

    #ajout des étiquettes selon les valeurs de l'âge pour les 10 classes d'âge
    for i in range(0,len(Y)):
        if Y[i]<10 : 
            Y.iloc[i]= str("0-9")
        elif Y[i]<20 :
            Y.iloc[i]= str("10-19")
        elif Y[i]<30 :
            Y.iloc[i]= str("20-29")
        elif Y[i]<40 :
           Y.iloc[i]= str("30-39")
        elif Y[i]<50 :
           Y.iloc[i]= str("40-49")
        elif Y[i]<60 :
            Y.iloc[i]= str("50-59")
        elif Y[i]<70 :
            Y.iloc[i]= str("60-69")
        elif Y[i]<80 :
            Y.iloc[i]= str("70-79")
        elif Y[i]<90 :
            Y.iloc[i]= str("80-89")
        elif Y[i]<100 :
            Y.iloc[i]= str("90-99")
        else : 
            Y.iloc[i]= str("100-109")

    #séparation en un ensemble d'apprentissage et de test (70%-30%)
    X_train, X_test, y_train, y_test = train_test_split(X, Y.astype(str), test_size = 0.3)

    feature_names = list(bd_final)
    print(feature_names)
    
    
    
    ##random forest avec le nombre d'arbres par défaut
    from sklearn.datasets import make_classification
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    #affichage du socre
    print("score sur ensemble de test RF avec classe",clf.score(X_test, y_test))
    
    #selection des variables intervenant dans la construction du modèle
    importance = clf.feature_importances_
    idx_third = importance.argsort()[-3]
    threshold = importance[idx_third] + 0.01
    
    idx_features = (-importance).argsort()[:5]
    name_features = np.array(feature_names)[idx_features]
    print('Selected features Random Forest : {}'.format(name_features))
    
    #affichage de la matrice de confusion, du score et de la table des classifications
    score1 = clf.score(X_test, y_test)
    predictions1 = clf.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test, predictions1))
    print(score1)
    print(classification_report(y_test,predictions1))
    
    #ajout du score à la liste
    score_RF.append(score1)
    
    
    ##reseau de neurones à 5 couches cachées de 50, 30, 30, 30 et 50 neurones
    print("Réseaux de neurones")
    clfe = MLPClassifier(hidden_layer_sizes=(50,30,30,30,50), max_iter=1000).fit(X_train, y_train)
    #affichage du score, de la matrice de confusion et de la table des classifications
    print(clfe)
    score = clfe.score(X_test, y_test)
    predictions = clfe.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test, predictions))
    print(score)
    print(classification_report(y_test,predictions))

    #ajout du score à la liste
    score_RN.append(score)

#affichage de la liste des scores
print(score_RF)
print(score_RN)

#calcul de la moyenne des scores
print('moyenne score RF '+str(mean(score_RF)))
print('moyenne score RN '+str(mean(score_RN)))

#plot du score de chaque itération et de la moyenne pour le random forest et le réseau de neurones
plt.plot(range(0,20), score_RF, 'navy', label='RF')
plt.hlines(mean(score_RF), 0, 20, 'royalblue', label='moyenne RF')
plt.plot(range(0,20), score_RN, 'darkorange', label="RN")
plt.hlines(mean(score_RN), 0, 20, 'moccasin', label='moyenne RN')
plt.xlabel('itération')
plt.ylabel('score de précision')
plt.legend()
 
