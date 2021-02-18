#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 10:52:44 2021

@author: roulletagathe
"""


import pandas as pds
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix


file = pds.read_csv("genus_level2.csv", index_col = 0, sep =",")
df  = pds.read_excel ("age_and_gender_jap.xlsx", index_col = 0)
df_inverse = np.transpose(df)
combined_csv = pds.concat([file, df_inverse])
df = combined_csv
df = df.drop(["gender", "BioSample No.", "number of high quality \nreads for 16S microbiota analysis " ])

bd_final = df.drop(["age"])
bd_final = np.transpose(bd_final)
X = bd_final 
y_non_sep = df.iloc[-1,:]
y_non_sep
Y = df.iloc[-1,:]

for i in range(0,len(Y)):
    if Y[i]<10 : 
        Y.iloc[i]= str("0-10")
    elif Y[i]>20 and Y[i]<60 :
        Y.iloc[i]= str("20-60")
    elif Y[i]>80: 
        Y.iloc[i]= str("80-109")
    else :
        Y.iloc[i]=pds.NaT

combined_csv = pds.concat([(X), Y], axis = 1)

combined_csv = combined_csv.dropna(subset=["age"])

size = len(combined_csv[(combined_csv["age"]=="0-10")])
sampler = np.random.randint(0, len(combined_csv[(combined_csv["age"]=="0-10")]), size = (size))
sampler
draws_0_3 = combined_csv[(combined_csv["age"]=="0-10")].take(sampler)
draws_0_3

sampler = np.random.randint(0, len(combined_csv[(combined_csv["age"]=="20-60")]), size = (size))
sampler
draws_15_40 = combined_csv[(combined_csv["age"]=="20-60")].take(sampler)
draws_15_40

sampler = np.random.randint(0, len(combined_csv[(combined_csv["age"]=="80-109")]), size = (size))
sampler
draws_80_109 = combined_csv[(combined_csv["age"]=="80-109")].take(sampler)
draws_80_109


bd_final = pds.concat([draws_0_3, draws_15_40, draws_80_109])


Y = bd_final["age"]
bd_final = np.transpose(bd_final)
X = bd_final.drop(["age"])
X = np.transpose(X)

liste_moyenne = []
for i in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
    print(" ")
    print("Résultat du random forest sur les données avec classe d'âge")
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    #a decommenter pour avois les features importants
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
# a decommenter pour avoir les matrices de confusion
    #print(confusion_matrix(y_test, predictions1))
    #print(score1)
    #print(classification_report(y_test,predictions1))
    liste_moyenne = liste_moyenne + [score]


#print(liste_moyenne)  
#print(confusion_matrix(y_test, predictions1))
#print(score)
#print(classification_report(y_test,predictions1))
#print(predictions1) 


#permet d'afficher les individus mal classés 
index_mal_classe = []
index = []
for i in range(0,len(y_test)):
    if y_test[i] != predictions1[i] : 
        index_mal_classe = index_mal_classe + [y_test.index[i]]
        index = index + [i]
        print(y_test.index[i],y_test[i],predictions1[i])
print(index_mal_classe)

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
y_non_sep = df.iloc[-1,:]
age = df.iloc[-1,:]

print("individu", "age", "Vraie classe", "Classe predite")
for i in range(0,len(index_mal_classe)-1):
    j = index[i]
    print(index_mal_classe[i], age.loc[index_mal_classe[i]], "   ", y_test[j], "   ",predictions1[j])


boxplotElements = plt.boxplot([liste_moyenne], sym = 'g*')
plt.gca().xaxis.set_ticklabels(["classes_distinctes"])
plt.title(' Résultats sur 10 répétitions de RF. Entrainement sur 70% des données.')
plt.suptitle("Précision pour des classes distinctes. 3 classes équilibrées : 0-10 ans/20-60 ans/80-109 ans. ")
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








