#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 21:12:31 2021

@author: roulletagathe
"""

import pandas as pds
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import scipy.stats as stat

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


score_RF = [] 
score_RN = []
for i in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(X, Y.astype(str), test_size = 0.3)
    
    ##random forest
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    #a decommenter pour avoir les features importantes    
    importance = clf.feature_importances_
    idx_features = (-importance).argsort()[:5]
    name_features = np.array(feature_names)[idx_features]

    score1 = clf.score(X_test, y_test)
    predictions1 = clf.predict(X_test)
   # a decommenter pour avoir les matrices de confusion
#    print(confusion_matrix(y_test, predictions1))
#    print(score1)
#    print(classification_report(y_test,predictions1))
    score_RF = score_RF + [score1]
    
    #reseau de neurones
    clfe = MLPClassifier(hidden_layer_sizes=(50,30,30,30,50), max_iter=1000).fit(X_train, y_train)
    score = clfe.score(X_test, y_test)
    predictions = clfe.predict(X_test)
    #a decommenter pour avoir les matrices de confusion
#    print(confusion_matrix(y_test, predictions))
#    print(score)
#    print(classification_report(y_test,predictions))
    score_RN = score_RN + [score]


#preparation des données
file = pds.read_csv("genus_level2.csv", index_col = 0, sep =",")
df  = pds.read_excel ("age_and_gender_jap.xlsx", index_col = 0)
div  = pds.read_excel ("diversity_genus_evenness_shannon.xlsx", index_col = 0)
df_inverse = np.transpose(df)
div_inverse = np.transpose(div)
combined_csv = pds.concat([file, div_inverse,df_inverse])
df = combined_csv
df = df.drop(["gender", "BioSample No.", "number of high quality \nreads for 16S microbiota analysis " ])
bd_final = df.drop(["age"])
bd_final = np.transpose(bd_final)
X = bd_final 
Y = df.iloc[-1,:]

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


score_RF_diversite = [] 
score_RN_diversite = []
for i in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(X, Y.astype(str), test_size = 0.3)
    
    ##random forest
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    #a decommenter pour avoir les variables intéressantes
#    importance = clf.feature_importances_
#    idx_features = (-importance).argsort()[:5]
#    name_features = np.array(feature_names)[idx_features]
#   print('Selected features Random Forest : {}'.format(name_features))
    score1 = clf.score(X_test, y_test)
    predictions1 = clf.predict(X_test)
    #a decommenter pour avoir les matrices de confusion
#    print(confusion_matrix(y_test, predictions1))
#    print(score1)
#    print(classification_report(y_test,predictions1))
    score_RF_diversite = score_RF_diversite + [score1]
    
    #reseau de neurones
    clfe = MLPClassifier(hidden_layer_sizes=(50,30,30,30,50), max_iter=1000).fit(X_train, y_train)
    score = clfe.score(X_test, y_test)
    predictions = clfe.predict(X_test)
    #a decommenter pour avoir les matrices de confusion
#    print(confusion_matrix(y_test, predictions))
#    print(score)
#    print(classification_report(y_test,predictions))
    score_RN_diversite = score_RN_diversite + [score]

boxplotElements = plt.boxplot([score_RF,
                                  score_RN, score_RF_diversite, score_RN_diversite], sym = 'g*')
plt.gca().xaxis.set_ticklabels(['RF', 'RN', "RF diversite", "RN diversite"])
plt.title(' Résultats sur 10 répétitions. Entrainement sur 70% des données.')
plt.suptitle('Ajout de données sur la diversité. Précision des random forests (RF) et des réseaux de neurones (RN) pour des âges en décade.')
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
print(stat.ttest_ind(score_RN,score_RN_diversite))
print(stat.ttest_ind(score_RF,score_RF_diversite))