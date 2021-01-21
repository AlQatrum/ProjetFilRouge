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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import KBinsDiscretizer

file = pds.read_csv("genus_level2.csv", index_col = 0, sep =",")

df  = pds.read_excel ("age_and_gender_jap.xlsx", index_col = 0)
#div  = pds.read_excel ("diversity_genus_evenness_shannon.xlsx", index_col = 0)
df_inverse = np.transpose(df)
#div_inverse = np.transpose(div)
#combined_csv = pds.concat([file, div_inverse,df_inverse])
combined_csv = pds.concat([file, df_inverse])
df = combined_csv

genre_identifie_ACP = ["g__Pseudoramibacter_Eubacterium", "g__Anaerofustis", "g__Corynebacterium", "g__Desulfovibrio", "g__Dorea", "g__Eggerthella", "g__Odoribacter", "g__Oscillospira",\
                     "g__Oxalobacter", "g__Parabacteroides", "g__Parvimonas", "g__Providencia",\
                     "g__Synergistes", "g__Turicibacter", "g__Anaerotruncus", "g__Fusobacterium",\
                      "g__Ruminococcus", "g__[Ruminococcus]", "g__Actinomyces", "g__Adlercreutzia",\
                      "g__Bifidobacterium", "g__Blautia", "g__Bulleidia", "g__Cloacibacillus", "g__Collinsella", "g__Escherichia", "g__Granulicatella", "g__Lachnospira", "g__Peptococcus",\
                       "g__Prevotella", "g__Propionibacterium", "g__Slackia"]

##A décommenter pour ne sélectionner que un genre
#df = np.transpose(df)
#print(df.shape)
#df = df.loc[df["gender"] == 'female']
#print(df.shape)
#df = np.transpose(df)


df = df.drop(["gender", "BioSample No.", "number of high quality \nreads for 16S microbiota analysis " ])

bd_final = df.drop(["age"])
bd_final = np.transpose(bd_final)
#bd_final = bd_final[genre_identifie_ACP] #on ne selectionne que les élements sélectionnés comme important dans l'ACP
X = bd_final #.astype(int)
Y = df.iloc[-1,:]



feature_names = list(bd_final)
print(feature_names)



X_train, X_test, y_train, y_test = train_test_split(X, Y.astype(str), test_size = 0.3)
#lasso
print(" ")
print("Résultat du lasso sur les données sans classe d'age")
clf = LassoCV(cv = 10).fit(X_train, y_train)
importance = np.abs(clf.coef_)


idx_third = importance.argsort()[-3]
threshold = importance[idx_third] + 0.01

idx_features = (-importance).argsort()[:5]
name_features = np.array(feature_names)[idx_features]
print('Selected features Lasso : {}'.format(name_features))

mean_square = clf.score(X_test,y_test)
print("Score Lasso ensemble test: ", mean_square)

print(" ")
print("Résultat du random forest sur les données sans classe d'âge")
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print("score sur ensemble de test RF sans classe",clf.score(X_test, y_test))


importance = clf.feature_importances_
idx_third = importance.argsort()[-3]
threshold = importance[idx_third] + 0.01

idx_features = (-importance).argsort()[:5]
name_features = np.array(feature_names)[idx_features]
print('Selected features Random Forest : {}'.format(name_features))
score = clf.score(X_test, y_test)
predictions = clf.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))
print(score)
print(classification_report(y_test,predictions))

#reseau de neurones
print("Réseaux de neurones sans classe")
clf = MLPClassifier(hidden_layer_sizes=(50,30,30,30,50), max_iter=1000).fit(X_train, y_train)

score = clf.score(X_test, y_test)
predictions = clf.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))
print(score)
print(classification_report(y_test,predictions))

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

X_train, X_test, y_train, y_test = train_test_split(X, Y.astype(str), test_size = 0.3)

feature_names = list(bd_final)
print(feature_names)




##random forest
from sklearn.datasets import make_classification
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
score1 = clf.score(X_test, y_test)
predictions1 = clf.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions1))
print(score1)
print(classification_report(y_test,predictions1))

#reseau de neurones
print("Réseaux de neurones")
clfe = MLPClassifier(hidden_layer_sizes=(50,30,30,30,50), max_iter=1000).fit(X_train, y_train)
print(clfe)
score = clfe.score(X_test, y_test)
predictions = clfe.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))
print(score)
print(classification_report(y_test,predictions))


#discretisation par 25

file = pds.read_csv("genus_level2.csv", index_col = 0, sep =",")
df  = pds.read_excel ("age_and_gender_jap.xlsx", index_col = 0)
#div  = pds.read_excel ("diversity_genus_evenness_shannon.xlsx", index_col = 0)
df_inverse = np.transpose(df)
#div_inverse = np.transpose(div)
#combined_csv = pds.concat([file, div_inverse,df_inverse])
combined_csv = pds.concat([file, df_inverse])
df = combined_csv
df = df.drop(["gender", "BioSample No.", "number of high quality \nreads for 16S microbiota analysis " ])

bd_final = df.drop(["age"])
bd_final = np.transpose(bd_final)
#bd_final = bd_final[genre_identifie_ACP] #on ne selectionne que les élements sélectionnés comme important dans l'ACP
X = bd_final
Y = df.iloc[-1,:]

liste_nb_groupes = []
liste_moyenne = []

enc = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy = 'quantile')
Y_array = np.asarray(Y).reshape(-1, 1)
Y_trans = enc.fit_transform(Y_array)
enc.bin_edges_[0]
X_train, X_test, y_train, y_test = train_test_split(X, Y_trans, test_size = 0.3)
print("Réseaux de neurones")
clfe = MLPClassifier(hidden_layer_sizes=(400,400,400,400), max_iter=1000).fit(X_train, y_train)
print(clfe)
score = clfe.score(X_test, y_test)
predictions = clfe.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))
print(score)
print(classification_report(y_test,predictions))
enc.bin_edges_[0]   


        
for i in range(2,23): 
    print(i)
    print("Resultats pour", i, "groupes")
    liste_nb_groupes = liste_nb_groupes + [i]
    enc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy = 'quantile')
    Y_array = np.asarray(Y).reshape(-1, 1)
    Y_trans = enc.fit_transform(Y_array)
    enc.bin_edges_[0]
    accuracy = 0
    compteur = 0
    for i in range(0,10):
        X_train, X_test, y_train, y_test = train_test_split(X, Y_trans, test_size = 0.3)
        print("Réseaux de neurones")
        clfe = MLPClassifier(hidden_layer_sizes=(400,400,400,400), max_iter=1000).fit(X_train, y_train)
        print(clfe)
        score = clfe.score(X_test, y_test)
        predictions = clfe.predict(X_test)
        from sklearn.metrics import classification_report, confusion_matrix
        print(confusion_matrix(y_test, predictions))
        print(score)
        print(classification_report(y_test,predictions))
        
        accuracy = accuracy + score
        compteur = compteur + 1
    print(accuracy/compteur)
    liste_moyenne = liste_moyenne + [accuracy/compteur]
len(liste_moyenne)
len(liste_nb_groupes)
print(liste_moyenne[10:31])
print(liste_nb_groupes[10:31])

plt.plot(liste_nb_groupes[10:31], liste_moyenne[10:31])
plt.show()