# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 17:25:20 2021

@author: hippo
"""

#bibliothèque utile
import pandas as pd
import numpy as np



#importation et preparation des données
df = pd.read_csv('C:/Users/hippo/Desktop/iodaa/Fil_Rouge/ProjetFilRouge/Data/InitialData/ratio_genre_trié_avec_age_transpose.csv',sep=',', header=0, index_col=0 )
df1 =  pd.read_csv('C:/Users/hippo/Desktop/iodaa/Fil_Rouge/ProjetFilRouge/Data/InitialData/ratio_genre_trié_avec_age_transpose.csv',sep=',', header=0, index_col=0 )
df.pop("age")
columns=df.columns.values 


#clustering des ratios


#importation des différentes méthode de clustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS

#parametrage de la méthode
kmeans = KMeans(n_clusters= 8, random_state=0)
#dbscan = DBSCAN(eps=10, min_samples=4)
#optics = OPTICS(min_samples=1)



#clustering dans les ratios
kmeans.fit(df)
#dbscan.fit(df)
#optics.fit(df)
cluster = kmeans.labels_




#recuperation des cluster
dico_clus={'0':[],'1':[],'2':[],'3':[],'4':[],
           '5':[],'6':[],'7':[],'8':[],'9':[],'-1':[]}
dico_rang={'0':[],'1':[],'2':[],'3':[],'4':[],
           '5':[],'6':[],'7':[],'8':[],'9':[],'-1':[]}

for i in range(len(cluster)):
    #recuperation des ages des clusters
    dico_clus[str(cluster[i])]=dico_clus[str(cluster[i])]+[int(df1["age"][i])] #"rang : "+str(i)
    #recuperation de tout les rang
    dico_rang[str(cluster[i])]=dico_rang[str(cluster[i])]+[i]

#calcul de la moyenne par cluster
list_moy_acp=[np.mean(dico_clus["0"]),
              np.mean(dico_clus["1"]),
              np.mean(dico_clus["2"]),
              np.mean(dico_clus["3"]),
              np.mean(dico_clus["4"]),
              np.mean(dico_clus["5"]),
              np.mean(dico_clus["6"]),
              np.mean(dico_clus["7"]),
              np.mean(dico_clus["8"]),
              np.mean(dico_clus["9"])
              ]

#calcul de l'ecart type par cluster
list_std_acp=[np.std(dico_clus["0"]),
              np.std(dico_clus["1"]),
              np.std(dico_clus["2"]),
              np.std(dico_clus["3"]),
              np.std(dico_clus["4"]),
              np.std(dico_clus["5"]),
              np.std(dico_clus["6"]),
              np.std(dico_clus["7"]),
              np.std(dico_clus["8"]),
              np.std(dico_clus["9"])
              ]

print(list_std_acp)
print(list_moy_acp)


