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



file = pds.read_csv("genus_level2.csv", index_col = 0, sep =",")

df  = pds.read_excel ("age_and_gender_jap.xlsx", index_col = 0)
df_inverse = np.transpose(df)
combined_csv = pds.concat([file, df_inverse])
df = combined_csv
df = df.drop(["gender", "BioSample No.", "number of high quality \nreads for 16S microbiota analysis " ])
df = np.transpose(df)

score_RF=[]
score_RN=[]

base_1=df[df["age"]<1.5]
base_2=df[df["age"]>=90]
base_3=df[(df["age"]>=1.5) & (df["age"]<90)]
bd_final=pds.concat([base_1, base_2, base_3])


X=bd_final.drop(["age"], axis=1)

bd_Y = np.transpose(bd_final)
Y=bd_Y.iloc[-1,:]

for i in range(0,len(Y)):
    if Y[i]<1.5 : 
        Y.iloc[i]= str("bebe")
    elif Y[i]>=85 :
        Y.iloc[i]= str("vieux")
    else:
        Y.iloc[i]= str("adulte")


for i in range(0,30):
    X_train, X_test, y_train, y_test = train_test_split(X, Y.astype(str), test_size = 0.3)
    
    feature_names = list(bd_final)
    print(feature_names)
    
    
    ##random forest
    from sklearn.datasets import make_classification
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
    
    score_RF.append(score1)
    
    #matrice de confusion
    sklearn.metrics.plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Blues)
    
    #age des mal classes
    bd_age=bd_final[["age"]]
    bd_age=bd_age.add_suffix('_reel')
    pred=pds.concat([y_test,bd_age],axis=1,join="inner")
    pred.insert(2, "classe_predite", predictions1, True)
    pred= pred.drop(pred[pred.age==pred.classe_predite].index)


#boxplot des précisions
boxplotElements = plt.boxplot([score_RF], sym = 'g*')
plt.title(' Résultats sur 30 répétitions. Entrainement sur 70% des données.')
plt.suptitle("Précision d'un RF sur trois classes : 0-1,5 1,5-85 et >85 ans.")
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

