# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 09:39:11 2021

@author: Hello
"""

import pandas as pds
import numpy as np
import random as rnd
from sklearn.model_selection import train_test_split

#importation des données
df  = pds.read_excel ("C:\\Users\\Hello\\Desktop\\IODAA\\fil_rouge\\age_and_gender_jap.xlsx", index_col = 0)
file = pds.read_csv("C:\\Users\\Hello\\Desktop\\IODAA\\fil_rouge\\genus_level2.csv", index_col = 0, sep =",")

#traitement des tableaux
df_inverse = np.transpose(df)
combined_csv = pds.concat([file, df_inverse])
df = combined_csv
df = df.drop(["gender", "BioSample No.", "number of high quality \nreads for 16S microbiota analysis " ])

df=np.transpose(df)
df_train, df_test = train_test_split(df, test_size = 0.3)
df=np.transpose(df_train.copy())

def gen_aleatoire(df):
    dfx=df.copy() #on fait une copie de df pour éviter le problème objet
    #on choisi aléatoirement les 200 individus qui vont être bruités
    for i in rnd.choices(range(np.shape(dfx)[1]), k=200):
        #on va bruités les valeurs autant de fois qu'on veut
        for k in [x for x in range(np.shape(dfx)[0]-2)  if df.iloc[x,i] != 0] :
            #on choisit la valeur du genre a bruité
            tirage_rang_1= rnd.choice([x for x in range(np.shape(dfx)[0]-2)  if df.iloc[x,i] != 0])
            #on fabrique le bruit
            tirage = rnd.uniform(-0.1,0.1)
            #on ajoute le bruit
            dfx.iloc[tirage_rang_1,i]=dfx.iloc[ tirage_rang_1,i]+tirage
            #dfx.iloc[k,i]=dfx.iloc[k,i]+tirage
            
            #afin que la somme des abondances relatives sur un indiviidus soit toujours égales a 1 on retires
            #le bruit a un autre individus choisis aléatoirement
            tirage_rang_2= rnd.choice([x for x in range(np.shape(dfx)[0]-2)  if dfx.iloc[x,i] != 0 and x!=tirage_rang_1])
            dfx.iloc[tirage_rang_2,i]=dfx.iloc[tirage_rang_2,i]-tirage
            
            #si un des genre bruité a une abondance négative à l'issue de l'opération, on fait marche arrière
            while dfx.iloc[tirage_rang_1,i]<0 or dfx.iloc[tirage_rang_2,i]<0:
                dfx.iloc[tirage_rang_2,i]=dfx.iloc[tirage_rang_2,i]+tirage
                dfx.iloc[tirage_rang_1,i]=dfx.iloc[ tirage_rang_1,i]-tirage
                tirage = rnd.uniform(-0.1,0.1)
                dfx.iloc[tirage_rang_1,i]=dfx.iloc[ tirage_rang_1,i]+tirage
                dfx.iloc[tirage_rang_2,i]=dfx.iloc[tirage_rang_2,i]-tirage
            
        #on bruite l'age en ajoutnt/retirant une année
        dfx.iloc[102,i] = dfx.iloc[102,i] + rnd.randint(-1,1)
    return(dfx)
 
    
#controle        
#for i in range(np.shape(df)[1]):
#    for j in range(np.shape(df)[0]):
#        if df.iloc[j,i]<0:
#            print(False)
 
nb_repet=6 
df1=gen_aleatoire(df)   
for k in range(nb_repet):             
    df2=gen_aleatoire(df)
    df1=pds.concat([df1, df2], axis=1)

df1.to_csv("C:\\Users\\Hello\\Desktop\\IODAA\\fil_rouge\\bd_final_genere_aleatoirement.csv")





"""test de la prédiction de l'age continue via un réseau de neurones/une random forest"""

from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
Y=df_test.copy().iloc[:,102]
Y=Y.astype('int')
X=df_test.copy().drop("age",axis=1)
X_train=np.transpose(df1.copy().drop("age",axis=0))
y_train =  df1.copy().iloc[102,:]
y_train=y_train.astype('int')
clf = MLPClassifier(hidden_layer_sizes=(400,400,400,400,400))
clf.fit(X_train, y_train)

prediction = clf.predict(X)
plt.scatter(prediction, Y)
plt.title("prediction par un réseau de neurones de l'age continue sur des données artificielles")
plt.xlabel("âge prédit")
plt.ylabel("âge réel")