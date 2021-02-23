# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 16:04:12 2020

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


"""calcul et plot divers de l'ACP"""

from sklearn.decomposition import PCA
pca = PCA(n_components=5) #n_components=5
principalComponents = pca.fit_transform(df)
pca_values=pca.components_



#Plot des individus
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#figure
plt.figure()
plt.figure(figsize=(10,10))

#legendes des axes
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)

plt.xlabel('Composante Principale - 1',fontsize=20)
plt.ylabel('Composante Principale  - 2',fontsize=20)
plt.title("ACP des microbiotes japonais",fontsize=20)

#plots des points
colors = cm.rainbow(np.linspace(0, 1, int(max(df1["age"]))+1 ) )
for k in range(len(principalComponents)):
    plt.scatter(principalComponents[k,0],principalComponents[k,1], c=colors[int(df1["age"][k])] )
       
#reglages    
plt.ylim(-60, -30)
plt.xlim(-90 ,-60)

cbar=plt.colorbar()
cbar.set_ticks(list(np.linspace(0, 1, 6 )))
cbar.set_ticklabels(list(np.linspace(0, int(max(df1["age"])),6 )))

plt.show()





#plot de la variance expliquée

#label
plt.xlabel('Composantes Principales',fontsize=10)
plt.ylabel('Part de la variance conservée',fontsize=10)
plt.title("Variance conservée par composante principale",fontsize=20)

#plot
plt.bar(range(len(pca.explained_variance_ratio_[0:9])),pca.explained_variance_ratio_[0:9])
plt.show()

#petite precision
print("on a : "+str(sum(pca.explained_variance_ratio_[0:9])*100)+"% de la variance expliquée dans les 10 premieère composantes ")





#plot des variables 
#avec sélection des groupes interessants

#differents groupes d'intérêts dans l'acp
valx=[]
valy=[]
valz=[]
for i in range(len(pca_values[0])):
    #valeurs issues de l'acp
    xi=pca_values[0][i]
    yi=pca_values[1][i]
    
    if 0.1<yi<0.2 and 0.035<xi<0.055:
        #recuperation du premier groupe
        valz=valz+[df.columns.values[i]]
        #plot du premier groupe
        plt.arrow(0,0, 
              dx=xi, dy=yi, color="green", 
              head_width=0.03, head_length=0.03, 
              length_includes_head=True)

    
    if 0.2<yi<0.5 and 0.02<xi<0.1:
        valx=valx+[df.columns.values[i]]
        plt.arrow(0,0, 
              dx=xi, dy=yi, color="red",
              head_width=0.03, head_length=0.03, 
              length_includes_head=True)

    if -0.1<yi<0 and 0.15<xi<0.5:
        valy=valy+[df.columns.values[i]]
        plt.arrow(0,0, 
              dx=xi, dy=yi, color="blue",
              head_width=0.03, head_length=0.03, 
              length_includes_head=True)

#setup du plot
plt.ylim(-0.25,0.6)
plt.xlim(-0.10,0.60)
plt.title('Détail du cercle de correlation')
plt.show()







"""preuve du gradient dans l'ACP"""

#selection de la fenetre
yup=-30
ydo=-54
xle=-84
xri=-60
#pas de découpe de la fenêtre de l'acp
pas=3

#matrice contenant la somme des ages
gradient=np.zeros([int(abs(yup-ydo)/pas),int(abs(xri-xle)/pas)])
#matrice contenant le nombre d'age comptabilisé
count_matrix=np.zeros([int(abs(yup-ydo)/pas),int(abs(xri-xle)/pas)])



for x in range(gradient.shape[1]):
     for y in range(gradient.shape[0]):
         for k in range(len(principalComponents)):
            if xri+pas*x<principalComponents[k,0]<xri+pas*(x+1) and ydo+pas*y<principalComponents[k,1]<yup+pas*(x+1):
                #si un point est dans la fenêtre il est comptabilisé
                #print(str(xri+pas*x)+"<"+str(principalComponents[k,0])+"<"+str(xri+pas*(x+1)))
                gradient[x,y]=gradient[x,y]+int(df1["age"][k])
                count_matrix[x,y]=count_matrix[x,y]+1


#on calcul la moyeenn de l'age dans chaque case
newgrad=np.zeros([int(abs(yup-ydo)/pas),int(abs(xri-xle)/pas)])
for x in range(gradient.shape[1]):
    for y in range(gradient.shape[0]):
       newgrad[x,y]=gradient[x,y]/count_matrix[x,y]
#on plot ce gradient   
plt.imshow(newgrad)

#reglages    
plt.colorbar()
plt.xlabel('Composante Principale 1',fontsize=10)
plt.ylabel('Composante Principale 2',fontsize=10)
plt.title("age moyen dans l'ACP",fontsize=20)

plt.show()
    







"""clustering dans l'acp"""


#importation des différentes méthode de clustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS

#parametrage de la méthode
kmeans = KMeans(n_clusters= 8, random_state=0)
#dbscan = DBSCAN(eps=10, min_samples=4)
#optics = OPTICS(min_samples=1)

#calcul de l'acp
pca = PCA() #n_components=5
principalComponents = pca.fit_transform(df)
pca_values=pca.components_


#◄il faut transformer le tableau de la pca qui contient 
#ligne : les dimension
#colonne : les obs

#coords=pd.DataFrame(principalComponents)
#coords=coords.transpose

principalComponents = pd.DataFrame(principalComponents)

#remove les outliers
principalComponents.drop([393,400,399,404])
#principalComponents.drop([399,404, 393, 403, 396])


#clustering dans l'acp
kmeans.fit(principalComponents)
#dbscan.fit(principalComponents)
#optics.fit(principalComponents)
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







"""clustering dasn les 21 ratio sélectionnés par l'acp"""
new_df=df[valx+valy+valz]


#importation des différentes méthode de clustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS

#parametrage de la méthode
kmeans = KMeans(n_clusters= 8, random_state=1)
#dbscan = DBSCAN(eps=10, min_samples=4)
#optics = OPTICS(min_samples=1)


#clustering dans l'acp
kmeans.fit(new_df)
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






"""plot interessant issues de l'acp"""

#barplot des moyennes de tout les ratios en fonction de l'age
plt.scatter(df1["age"],np.mean(new_df, axis= 1))
plt.ylim(0,10)
plt.xlim(0,100)
plt.show()


#matrice de correlation avec l'age des variables selectionnées
import seaborn as sn
new_df_age=new_df
new_df_age["age"]=df1['age']
corrMatrix=new_df_age.corr()
sn.heatmap(corrMatrix, annot=False, xticklabels=False, yticklabels=False)
plt.title("matrice de corrélation de l'age avec les 21 variables ressortie de l'acp",fontsize=10)
plt.show()



pca = PCA() #n_components=5
principalComponents = pca.fit_transform(df)
pca_values=pca.components_





"""plot de l'age en fonction des (log de) première dimension"""

import math

x_=[]
y_=[]
for k in range(len(principalComponents)):
    if not math.isnan(np.log(principalComponents[k,0])):
        x_ = x_ + [ np.log(principalComponents[k,0]) ]
        y_ = y_ + [ int(df1["age"][k]) ]    
    plt.scatter(np.log(principalComponents[k,0]),int(df1["age"][k]), c= "blue", marker="+" )

plt.ylim(0,100)
plt.xlim(0,11)
plt.xlabel('Log de la première composante',fontsize=10)
plt.ylabel('Age',fontsize=10)
plt.title("age des personne en fonction du log de la première composante (ratio)",fontsize=10)

#plot de la regression lineaire
m, b = np.polyfit(x_, y_, 1)
plt.plot(x_, m*np.array(x_)+ b, color ="g")

plt.show()

   




"""Regression linéaire sur composante principales"""

X=df
Y=df1["age"]


from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score


def pcr(X,y,pc):
    ''' Principal Component Regression'''
    # ACP sur les pc dimension choisies
    pca = PCA(n_components=pc)
    Xreg = pca.fit_transform(X)
    
    # regression linéaire
    regr = linear_model.LinearRegression()
    
    # Fit
    regr.fit(Xreg, y)
    
    # Calibration
    y_c = regr.predict(Xreg)
    # Cross-validation
    y_cv = cross_val_predict(regr, Xreg, y, cv=10)
    
    # R² isssue de la calibration et de la cross validation
    score_c = r2_score(y, y_c)
    score_cv = r2_score(y, y_cv)
    
    # EQM pour la calibration et la cv
    mse_c = mean_squared_error(y, y_c)
    mse_cv = mean_squared_error(y, y_cv)
    
    return(y_cv, score_c, score_cv, mse_c, mse_cv)

pcr(X,Y, 150)

x=range(0, 451)
y=[]
#for k in range(2,453):
#    y=y+[pcr(X,Y,k)[1]]

plt.plot(x,y)
plt.title("R² en fonction de du nombre de composantes utilisées")
plt.xlabel("nombre de composantes principales")
plt.ylabel("R²")
plt.show()





"""regression linéaire sur les variables sélectionnés par acp"""

Xreg=df[valx+valy+valz]
Y=df1["age"]
y=Y

# regression linéaire
regr = linear_model.LinearRegression()
    
# Fit
regr.fit(Xreg, y)

# Calibration
y_c = regr.predict(Xreg)
# Cross-validation
y_cv = cross_val_predict(regr, Xreg, y, cv=10)

# R² isssue de la calibration et de la cross validation
score_c = r2_score(y, y_c)
score_cv = r2_score(y, y_cv)

# EQM pour la calibration et la cv
mse_c = mean_squared_error(y, y_c)
mse_cv = mean_squared_error(y, y_cv)

print(score_c, score_cv, mse_c, mse_cv)