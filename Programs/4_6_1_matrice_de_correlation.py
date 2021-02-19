# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:48:41 2021

@author: Hello
"""


#importation des données

#bibliothèque importantes
import pandas as pd
import matplotlib.pyplot as plt

#importation des ratios de genres
df=pd.read_csv('C:\\Users\\Hello\\Desktop\\IODAA\\fil_rouge\\ratio_genre_trié_avec_age_transpose.csv', sep=',', header=0 )
df=df.drop(columns=["Unnamed: 0"])

#le calcul de la matrice de corrélation est beaucoup trop long et prends trop de ram (16GB insuffisant) 
#coorelation=df['age'].corr()


#calcul de la correlation avec l'age
correlation=df.corrwith(df["age"], axis=0)
correlation=correlation.sort_values(ascending=False)

#selection des 10 variables avec la plus grande correlation
corr_genus=correlation[1:10]
min_corr_genus=correlation[len(correlation)-10:len(correlation)] 


#barplot de 10 ùmeilleures correlation0
plt.bar(range(len(corr_genus)), corr_genus)
plt.title("10 plus grandes correlations")
plt.xticks(rotation=-45, ha="right")
plt.show()

#les ratios avec les plus grandes corrélations sont essentiellement des ratio de genres identiques.
#cette approche des  ratios de genres ne semble donc pas très efficace.
#de plus les meilleures correlation sont assez faibles 0.4.


