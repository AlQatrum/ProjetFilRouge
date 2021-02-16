# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 15:13:20 2021

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

file = pds.read_csv("table_age_1_sex_espece_diversite_VF.csv", index_col = 0, sep =",")
df = np.transpose(file)
df = df.drop(["Subject", "age", "gender", "shannon_index", "evenness_index"])


bd_final = df.drop(["age2"])
bd_final = np.transpose(bd_final)
#bd_final = bd_final[genre_identifie_ACP] #on ne selectionne que les élements sélectionnés comme important dans l'ACP
X = bd_final #.astype(int)
Y = df.iloc[0,:]



feature_names = list(bd_final)
print(feature_names)



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


RF_score=[]
RN_score=[]



for i in range(0,30):
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
    RF_score.append(score1)
    
    
    
    
    
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
    RN_score.append(score)
    
  
    
#boxplot des précisions
boxplotElements = plt.boxplot([RF_score, RN_score], sym = 'g*')
plt.gca().xaxis.set_ticklabels(['RF', 'RN'])
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
