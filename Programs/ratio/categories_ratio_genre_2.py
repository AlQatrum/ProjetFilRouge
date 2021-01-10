# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 14:38:36 2021

@author: hippo
"""


import pandas as pd

df = pd.read_csv('C:/Users/hippo/Desktop/iodaa/Fil_Rouge/ProjetFilRouge/Data/InitialData/ratio_genre_trié_avec_age_transpose.csv',sep=',', header=0, index_col=0 )
df1 = df.astype({"age": str})


for k in range(len(df["age"])):
    if df["age"][k]<7 :
        df1["age"][k]="child"
    elif 7<=df["age"][k]<75:
        df1["age"][k]="adult"
    else:
        df1["age"][k]="elderly"


df1.to_csv("C:/Users/hippo/Desktop/iodaa/Fil_Rouge/ProjetFilRouge/Data/InitialData/ratio_genre_classé_2.csv")
