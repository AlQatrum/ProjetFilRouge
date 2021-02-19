# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 14:20:42 2021

@author: Hello
"""

#bilbliothèque interessante
import pandas as pds
import numpy as np
import sklearn 
print(sklearn.__version__)

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#importation des données
df_gen = pds.read_csv("C:\\Users\\Hello\\Desktop\\IODAA\\fil_rouge\\bd_final_genere_aleatoirement.csv", index_col = 0, sep =",")
Y=df_gen.iloc[-1,:]
X=np.transpose(df_gen.copy().drop( ["age"]))



#label des classes discontinues
for i in range(0,len(Y)):
    if Y[i]<10 : 
        Y.iloc[i]= str("0-10")
    elif Y[i]>20 and Y[i]<60 :
        Y.iloc[i]= str("20-60")
    elif Y[i]>80: 
        Y.iloc[i]= str("80-104")
    else:
        Y.iloc[i]=pds.NaT
    



#creation des classes discontinues 

combined_csv = pds.concat([(X), Y], axis = 1)
combined_csv = combined_csv.dropna(subset=["age"]) #on combine les X avec les Y selectionnés et on supprime les na

#classe 0-10
size = len(combined_csv[(combined_csv["age"]=="0-10")]) #la tailles des échantillons, les enfants sont limitants
sampler = np.random.randint(0, len(combined_csv[(combined_csv["age"]=="0-10")]), size = (size))
draws_0_3 = combined_csv[(combined_csv["age"]=="0-10")].take(sampler)

#classe 20-60
sampler = np.random.randint(0, len(combined_csv[(combined_csv["age"]=="20-60")]), size = (size))
draws_15_40 = combined_csv[(combined_csv["age"]=="20-60")].take(sampler)

#classe 80-104
sampler = np.random.randint(0, len(combined_csv[(combined_csv["age"]=="80-104")]), size = (size))
draws_80_104 = combined_csv[(combined_csv["age"]=="80-104")].take(sampler)

bd_final = pds.concat([draws_0_3, draws_15_40, draws_80_104])


#données d'entrainement
Y = bd_final["age"]
bd_final = np.transpose(bd_final)
X = bd_final.drop(["age"])
X = np.transpose(X)

#print(X,Y) #controle



"""entrainement du modèle"""

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
clf = RandomForestClassifier(n_estimators=100, n_jobs= 4) # !!! BEWARE : parallel !!!
clf.fit(X_train, y_train)



"""plot de la confusion matrix sur les données générées artificiellement"""

predictions1 = clf.predict(X_test) #on calccul la prediction sur l'ensemble de test
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
matrice_conf=confusion_matrix(y_test, predictions1)
print(matrice_conf)
plot_confusion_matrix( clf, X_test, y_test, cmap=plt.cm.Blues) #on plot la matrice de conf
plt.title("Matrice de confusion d'une RF sur classes discontinues\n(test sur les données artificielles)")
plt.show()











"""test sur le jeu de données originel"""


#importation des données
df  = pds.read_excel ("C:\\Users\\Hello\\Desktop\\IODAA\\fil_rouge\\age_and_gender_jap.xlsx", index_col = 0)
file = pds.read_csv("C:\\Users\\Hello\\Desktop\\IODAA\\fil_rouge\\genus_level2.csv", index_col = 0, sep =",")
#file = np.transpose(file)
#file= file.drop(["age"])

#traitement des tableaux
df_inverse = np.transpose(df)
combined_csv = pds.concat([file, df_inverse])
df = combined_csv

df = df.drop(["gender", "BioSample No.", "number of high quality \nreads for 16S microbiota analysis " ])
bd_final = df.drop(["age"])
bd_final = np.transpose(bd_final)

#tableau finaux pour sklearn
X = bd_final #.astype(int)
Y= df.iloc[-1,:]



#label des classes discontinues

for i in range(0,len(Y)):
    if Y[i]<10 : 
        Y.iloc[i]= str("0-10")
    elif Y[i]>20 and Y[i]<60 :
        Y.iloc[i]= str("20-60")
    elif Y[i]>80: 
        Y.iloc[i]= str("80-104")
    else:
        Y.iloc[i]=pds.NaT
    


#creation des classes discontinues 

combined_csv = pds.concat([(X), Y], axis = 1)
combined_csv = combined_csv.dropna(subset=["age"]) #on combine les X avec les Y selectionnés et on supprime les na

#classe 0-10
size = len(combined_csv[(combined_csv["age"]=="0-10")]) #la tailles des échantillons, les enfants sont limitants
sampler = np.random.randint(0, len(combined_csv[(combined_csv["age"]=="0-10")]), size = (size))
draws_0_3 = combined_csv[(combined_csv["age"]=="0-10")].take(sampler)

#classe 20-60
sampler = np.random.randint(0, len(combined_csv[(combined_csv["age"]=="20-60")]), size = (size))
draws_15_40 = combined_csv[(combined_csv["age"]=="20-60")].take(sampler)

#classe 80-104
sampler = np.random.randint(0, len(combined_csv[(combined_csv["age"]=="80-104")]), size = (size))
draws_80_104 = combined_csv[(combined_csv["age"]=="80-104")].take(sampler)

bd_final = pds.concat([draws_0_3, draws_15_40, draws_80_104])


#données d'entrainement
Y = bd_final["age"]
bd_final = np.transpose(bd_final)
X = bd_final.drop(["age"])
X = np.transpose(X)

#print(X,Y) #controle

predictions1 = clf.predict(X) #on calccul la prediction sur l'ensemble de test
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
matrice_conf=confusion_matrix(Y, predictions1)
print(matrice_conf)
plot_confusion_matrix( clf, X, Y, cmap=plt.cm.Blues) #on plot la matrice de conf
plt.title("Matrice de confusion d'une RF sur classes discontinues\n(test sur le jeu de données originel)")
plt.show()

clf.score(X,Y)
