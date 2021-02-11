# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 18:18:19 2020

@author: hippo
"""
"""création des catégories de ratios"""
import pandas as pd

#importation des données
df = pd.read_csv('C:/Users/hippo/Desktop/iodaa/Fil_Rouge/ProjetFilRouge/Data/InitialData/ratio_genre_trié_avec_age_transpose.csv',sep=',', header=0, index_col=0 )
df1 = df.astype({"age": str})

nchild=0
nadult=0
nelder=0
df1index=df1.index
#on remplie chaque classe avec 59 individus
for k in range(len(df["age"])):
    if df["age"][k]<7 and nchild<59:
        df1["age"][k]="child"
        nchild=nchild+1
    elif 7<=df["age"][k]<75 and nadult<59:
        df1["age"][k]="adult"
        nadult=nadult+1
    elif nelder<59:
        df1["age"][k]="elderly"
        nelder=nelder+1
print(nchild, nadult, nelder)

df2 = df1[df1.age == "adult"]
df3 = df1[df1.age == "child"]
df4 = df1[df1.age == "elderly"]
frames =[df2, df3 , df4]
result= pd.concat(frames)

#on enregistre le résultat
result.to_csv("C:/Users/hippo/Desktop/iodaa/Fil_Rouge/ProjetFilRouge/Data/InitialData/ratio_genre_classé_1.csv")









