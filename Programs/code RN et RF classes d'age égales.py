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

file = pds.read_csv("genus_level2.csv", index_col = 0, sep =",")

df  = pds.read_excel ("age_and_gender_jap.xlsx", index_col = 0)
df_inverse = np.transpose(df)
combined_csv = pds.concat([file, df_inverse])
df = combined_csv
df = df.drop(["gender", "BioSample No.", "number of high quality \nreads for 16S microbiota analysis " ])
df = np.transpose(df)

score_RF=[]
score_RN=[]

tour=0

for tour in range(0,10):
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

    bd_final=pds.concat([base_1, base_2, base_3, base_4, base_5, base_6, base_7, base_8, base_9, base_10])

    X=bd_final.drop(["age"], axis=1)
    
    bd_Y = np.transpose(bd_final)
    Y=bd_Y.iloc[-1,:]

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

    score_RN.append(score)
    
print(score_RF)
print(score_RN)

print('moyenne score RF '+str(mean(score_RF)))
print('moyenne score RN '+str(mean(score_RN)))
 