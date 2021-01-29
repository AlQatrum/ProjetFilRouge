#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 10:52:44 2021

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
from sklearn import preprocessing
#from sklearn.metrics import ConfusionMatrixDisplay

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

for i in range(0,len(Y)):
    if Y[i]<10 : 
        Y.iloc[i]= str("0-10")
    elif Y[i]>20 and Y[i]<60 :
        Y.iloc[i]= str("20-60")
    elif Y[i]>80: 
        Y.iloc[i]= str("80-109")
    else :
        Y.iloc[i]=pds.NaT

print(Y)



#div_inverse = np.transpose(div)
#combined_csv = pds.concat([file, div_inverse,df_inverse])
combined_csv = pds.concat([(X), Y], axis = 1)
type(X)
combined_csv
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



print(X,Y)

liste_moyenne = []
for i in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
    from sklearn.datasets import make_classification
    print(" ")
    print("Résultat du random forest sur les données avec classe d'âge")
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
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
    from sklearn.metrics import classification_report, confusion_matrix
    #print(confusion_matrix(y_test, predictions1))
    #print(score1)
    #print(classification_report(y_test,predictions1))
    liste_moyenne = liste_moyenne + [score]

print(liste_moyenne)  
print(confusion_matrix(y_test, predictions1))
print(score)
print(classification_report(y_test,predictions1))

#titles_options = [("Confusion matrix, without normalization", None),
#                  ("Normalized confusion matrix", 'true')]
#for title, normalize in titles_options:
#    disp = plot_confusion_matrix(clf, X_test, y_test,
#                                 display_labels=class_names,
#                                 cmap=plt.cm.Blues,
#                                 normalize=normalize)
#    disp.ax_.set_title(title)
#
#    print(title)
#    print(disp.confusion_matrix)

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


import matplotlib.pyplot as plt

# VARIABLE QUALITATIVE
# Diagramme en secteurs
Y.value_counts(normalize=False).plot(kind='pie')
# Cette ligne assure que le pie chart est un cercle plutôt qu'une éllipse
plt.axis('equal') 
plt.show() # Affiche le graphique

Y.value_counts()
# Diagramme en tuyaux d'orgues
Y.value_counts(normalize=False).plot(kind='bar')
plt.show()

# VARIABLE QUANTITATIVE
# Diagramme en bâtons
Y.value_counts(normalize=False).plot(kind='bar',width=0.1)
plt.show()






