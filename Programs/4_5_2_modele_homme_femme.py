#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 11:10:37 2021

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import KBinsDiscretizer
import scipy.stats as stat
from sklearn.metrics import classification_report, confusion_matrix



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
    
#construction des classes d'âge
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


#Répétition des algorithmes 10 fois
score_RF = [] 
score_RN = []
for i in range(0,10):
    #creation d'un jeu de test et d'entrainement
    X_train, X_test, y_train, y_test = train_test_split(X, Y.astype(str), test_size = 0.3)
    
    ##random forest
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    score1 = clf.score(X_test, y_test)
    predictions1 = clf.predict(X_test)
    
    #a decommenter pour avoir les matrices de confusion
    #print(confusion_matrix(y_test, predictions1))
    #print(score1)
    #print(classification_report(y_test,predictions1))
    score_RF = score_RF + [score1]
    
    #reseau de neurones
    clfe = MLPClassifier(hidden_layer_sizes=(50,30,30,30,50), max_iter=1000).fit(X_train, y_train)
    score = clfe.score(X_test, y_test)
    predictions = clfe.predict(X_test)
    
    #a decommenter pour avoir les matrices de confusion
    #print(confusion_matrix(y_test, predictions))
    #print(score)
    #print(classification_report(y_test,predictions))
    score_RN = score_RN + [score]

#preparation du modèle pour les femmes
#preparation des données
file = pds.read_csv("genus_level2.csv", index_col = 0, sep =",")
df  = pds.read_excel ("age_and_gender_jap.xlsx", index_col = 0)
df_inverse = np.transpose(df)
combined_csv = pds.concat([file, df_inverse])
df = combined_csv

#on commence par le modèle pour les femmes
df = np.transpose(df)
df_femme = df.loc[df["gender"] == 'female']
df_femme = np.transpose(df_femme)

df_femme = df_femme.drop(["gender", "BioSample No.", "number of high quality \nreads for 16S microbiota analysis " ])
bd_final = df_femme.drop(["age"])
bd_final = np.transpose(bd_final)
X = bd_final 
Y = df_femme.iloc[-1,:]

#construction des classes d'âge
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
        
score_RF_femme = [] 
score_RN_femme = []
for i in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(X, Y.astype(str), test_size = 0.3)
    
    ##random forest
    from sklearn.datasets import make_classification
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    # a decommenter pour avoir les features importantes
#    print("score sur ensemble de test RF avec classe",clf.score(X_test, y_test))
#    importance = clf.feature_importances_  
#    idx_features = (-importance).argsort()[:5]
#    name_features = np.array(feature_names)[idx_features]
#    print('Selected features Random Forest : {}'.format(name_features))
    
    score1 = clf.score(X_test, y_test)
    predictions1 = clf.predict(X_test)
    
    #a decommenter pour avoir les matrices de confusion
#    print(confusion_matrix(y_test, predictions1))
#    print(score1)
#    print(classification_report(y_test,predictions1))
    score_RF_femme = score_RF_femme + [score1]
    
    #reseau de neurones
    clfe = MLPClassifier(hidden_layer_sizes=(50,30,30,30,50), max_iter=1000).fit(X_train, y_train)
    score = clfe.score(X_test, y_test)
    predictions = clfe.predict(X_test)
    #a decommenter pour avoir les matrices de confusion
#    print(confusion_matrix(y_test, predictions))
#    print(score)
#    print(classification_report(y_test,predictions))
    score_RN_femme = score_RN_femme + [score]


#on fait de même pour les hommes
file = pds.read_csv("genus_level2.csv", index_col = 0, sep =",")
df  = pds.read_excel ("age_and_gender_jap.xlsx", index_col = 0)
df_inverse = np.transpose(df)
combined_csv = pds.concat([file, df_inverse])
df = combined_csv
df = np.transpose(df)
df_homme = df.loc[df["gender"] == 'male']
df_homme = np.transpose(df_homme)
df_homme = df_homme.drop(["gender", "BioSample No.", "number of high quality \nreads for 16S microbiota analysis " ])
bd_final = df_homme.drop(["age"])
bd_final = np.transpose(bd_final)
X = bd_final
Y = df_homme.iloc[-1,:]

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

score_RF_homme = [] 
score_RN_homme = []
for i in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(X, Y.astype(str), test_size = 0.3)
    
    
    ##random forest
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    #a decommenter pour avoir les features importantes
#    print("score sur ensemble de test RF avec classe",clf.score(X_test, y_test))
#    importance = clf.feature_importances_
#    idx_features = (-importance).argsort()[:5]
#    name_features = np.array(feature_names)[idx_features]
#    print('Selected features Random Forest : {}'.format(name_features))
    
    score1 = clf.score(X_test, y_test)
    predictions1 = clf.predict(X_test)
    #a decommenter pour avoir les matrices de confusion
#    print(confusion_matrix(y_test, predictions1))
#    print(score1)
#    print(classification_report(y_test,predictions1))
    score_RF_homme = score_RF_homme + [score1]
    
    #reseau de neurones
    clfe = MLPClassifier(hidden_layer_sizes=(50,30,30,30,50), max_iter=1000).fit(X_train, y_train)
    score = clfe.score(X_test, y_test)
    predictions = clfe.predict(X_test)
    #a decommenter pour avoir les matrices de confusion
#    print(confusion_matrix(y_test, predictions))
#    print(score)
#    print(classification_report(y_test,predictions))
    score_RN_homme = score_RN_homme + [score]


boxplotElements = plt.boxplot([score_RF,
                                  score_RN, score_RF_femme, score_RN_femme, score_RF_homme, score_RN_homme], sym = 'g*')
plt.gca().xaxis.set_ticklabels(['RF', 'RN', 'RF femme', 'RN femme',"RF homme", "RN homme"])
plt.title(' Résultats sur 10 répétitions. Entrainement sur 70% des données.')
plt.suptitle('Comparaison Homme/Femme. Précision des random forests (RF) et des réseaux de neurones (RN) pour des âges en décade.')
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

print(stat.ttest_ind(score_RF_femme,score_RF_homme)) # test de student différence homme/femme random forest, différence significative
print(stat.ttest_ind(score_RN_femme,score_RN_homme)) # test de student différence homme/femme réseaux neurones, différence significative
