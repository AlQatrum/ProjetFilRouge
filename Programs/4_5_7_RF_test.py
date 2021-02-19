# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 10:50:16 2021

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

X_interm = bd_final
Y_interm = df.iloc[-1,:].copy()
Y_true_label = df.iloc[-1,:].copy()



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
    


#label des classes intermediaires

for i in range(0,len(Y_interm)):
    if  Y_interm[i]>=10 and Y_interm[i]<15 :
        Y_interm.iloc[i]= str("0-10")
        Y_true_label.iloc[i]= str("10-15")
    elif  Y_interm[i]>=15 and Y_interm[i]<=20 :
        Y_interm.iloc[i]= str("20-60")
        Y_true_label.iloc[i]= str("15-20")
    elif   Y_interm[i]>=60 and  Y_interm[i]<70 :
        Y_interm.iloc[i]= str("20-60")
        Y_true_label.iloc[i]= str("60-70")
    elif   Y_interm[i]>=70 and  Y_interm[i]<=80 :
        Y_interm.iloc[i]= str("80-104")
        Y_true_label.iloc[i]= str("70-80")
    else:
        Y_interm.iloc[i]=pds.NaT
        Y_true_label.iloc[i]=pds.NaT
        
#print(Y) #controle





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
A=set(sampler) #on passe au ensemble pour récupérer les échantillon non utilisé par le sampler
B=set( range( 1,len(combined_csv[(combined_csv["age"]=="20-60")] ) ) )
complementary_sampler= list(B-A)
reject_15_40 = combined_csv[(combined_csv["age"]=="20-60")].take(complementary_sampler)

#classe 80-104
sampler = np.random.randint(0, len(combined_csv[(combined_csv["age"]=="80-104")]), size = (size))
draws_80_104 = combined_csv[(combined_csv["age"]=="80-104")].take(sampler)
A=set(sampler) #on passe au ensemble pour récupérer les échantillon non utilisé par le sampler
B=set( range( 1,len(combined_csv[(combined_csv["age"]=="80-104")] ) ) )
complementary_sampler= list(B-A)
reject_80_104 = combined_csv[(combined_csv["age"]=="80-104")].take(complementary_sampler)

bd_final = pds.concat([draws_0_3, draws_15_40, draws_80_104])
bdsupp = pds.concat([reject_15_40, reject_80_104])

#données d'entrainement
Y = bd_final["age"]
bd_final = np.transpose(bd_final)
X = bd_final.drop(["age"])
X = np.transpose(X)

#print(X,Y) #controle

#données de rab
Y_prim = bdsupp["age"]
bdsupp = np.transpose(bdsupp)
X_prim = bdsupp.drop(["age"])
X_prim = np.transpose(X_prim)



#creation des interclasses

combined_csv = pds.concat([(X_interm), Y_interm], axis = 1)
combined_csv = combined_csv.dropna(subset=["age"]) #on combine les X avec les Y selectionnés

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

bd_final = combined_csv

Y_interm = bd_final["age"]
bd_final = np.transpose(bd_final)
X_interm = bd_final.drop(["age"])
X_interm = np.transpose(X_interm)






"""plot de la precision en fonction du nombre d'arbres dans la RF"""


#paramètres nombre d'arbres différents 
n_arbre=[100,200,300,400,500]
moy_precision=[]
std=[]
for j in n_arbre :
    liste_moyenne = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3) #creation des jeux de données train et test
        clf = RandomForestClassifier(n_estimators=j, n_jobs= 4) # !!! BEWARE : parallel !!!
        #clf = tree.DecisionTreeClassifier() #arbre de précision simple
        clf.fit(X_train, y_train) #entrainement de la RF
        score = clf.score(X_test, y_test) #calcul de la précision sur l'ensemble de test
        liste_moyenne = liste_moyenne + [score] 
    moy_precision= moy_precision+ [np.mean(liste_moyenne)] #calcul de a precision moyenne pour les 50 runs
    std=std+[np.std(liste_moyenne)] #idem pour la sd
    
#plot    
moyenne = plt.plot(n_arbre, moy_precision, label='moyenne')
std=np.array(std)
moy_precision=np.array(moy_precision)
ecart_type = plt.fill_between(n_arbre, moy_precision+std,  moy_precision-std, facecolor='blue', alpha=0.5, label="écart-type")
plt.legend()
plt.title("moyenne de la précision en fonction du nombre d'arbres ("+str(i+1)+ " repétitions)")
plt.xlabel("nombre d'arbres")
plt.ylabel("précision")
plt.show()
  







clf = RandomForestClassifier(n_estimators=100, n_jobs= 4) # !!! BEWARE : parallel !!!
clf.fit(X_train, y_train)



"""plot de la confusion matrix"""

predictions1 = clf.predict(X_test) #on calcul la prediction sur l'ensemble de test
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
matrice_conf=confusion_matrix(y_test, predictions1)
print(matrice_conf)
plot_confusion_matrix( clf, X_test, y_test, cmap=plt.cm.Blues) #on plot la matrice de conf
plt.title("Matrice de confusion d'une RF sur classes discontinues")
plt.show()



"""test sur les  données supplémentaires"""

predictions1 = clf.predict(X_prim)
matrice_conf=confusion_matrix(Y_prim, predictions1)
score = clf.score(X_prim, Y_prim)
plot_confusion_matrix( clf, X_prim, Y_prim, cmap=plt.cm.Blues)
plt.title("Matrice de confusion d'une RF sur classes discontinues (rab)")
plt.show()







"""test sur les données intermédiaires"""

predictions1 = clf.predict(X_interm)
matrice_conf=confusion_matrix(Y_interm, predictions1)
score = clf.score(X_interm, Y_interm)
plot_confusion_matrix( clf, X_interm, Y_interm, cmap=plt.cm.Blues)
plt.title("Matrice de confusion d'une RF sur classes discontinues (intermédiaires)")
plt.show()

#plot de la vraie matrice de conf
Y_true_label=Y_true_label.dropna()
Y_true_label=np.array(Y_true_label)
matrice_conf=np.zeros([4,3])
for k in range( len(predictions1) ): #on rempli la matrice avec les valeur pour les interclasses
    if predictions1[k]=="10-20" and Y_true_label[k]=="10-15":
        matrice_conf[0,0]=matrice_conf[0,0]+1
    elif predictions1[k]=="20-60" and Y_true_label[k]=="10-15":
        matrice_conf[0,1]=matrice_conf[0,1]+1
    elif predictions1[k]=="80-104" and Y_true_label[k]=="10-15":
        matrice_conf[0,2]=matrice_conf[0,2]+1
        
    elif predictions1[k]=="10-20" and Y_true_label[k]=="15-20":
        matrice_conf[1,0]=matrice_conf[1,0]+1
    elif predictions1[k]=="20-60" and Y_true_label[k]=="15-20":
        matrice_conf[1,1]=matrice_conf[1,1]+1
    elif predictions1[k]=="80-104" and Y_true_label[k]=="15-20":
        matrice_conf[1,2]=matrice_conf[1,2]+1
        
    elif predictions1[k]=="10-20" and Y_true_label[k]=="60-70":
        matrice_conf[2,0]=matrice_conf[2,0]+1
    elif predictions1[k]=="20-60" and Y_true_label[k]=="60-70":
        matrice_conf[2,1]=matrice_conf[2,1]+1
    elif predictions1[k]=="80-104" and Y_true_label[k]=="60-70":
        matrice_conf[2,2]=matrice_conf[2,2]+1
        
    elif predictions1[k]=="10-20" and Y_true_label[k]=="70-80":
        matrice_conf[3,0]=matrice_conf[3,0]+1
    elif predictions1[k]=="20-60" and Y_true_label[k]=="70-80":
        matrice_conf[3,1]=matrice_conf[3,1]+1
    elif predictions1[k]=="80-104" and Y_true_label[k]=="70-80":
        matrice_conf[3,2]=matrice_conf[3,2]+1
        
#on plot la matice construite via seaborn
import seaborn as sn
sn.heatmap(matrice_conf,cmap=plt.cm.Blues,annot=True, annot_kws={"size": 10},linecolor='black', 
           linewidths=.3, xticklabels=["0-10",'20-60','80-104'], yticklabels=["10-15",'15-20','60-70','70-80'] )
plt.title("matrice de confusion sur les données intermédiaires")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()




"""test sur abre de decision simple"""

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10), dpi=400)
tree.plot_tree(clf,feature_names = X.columns )

predictions1 = clf.predict(X_test)
matrice_conf=confusion_matrix(y_test, predictions1)
plot_confusion_matrix( clf, X_test, y_test, cmap=plt.cm.Blues)
plt.title("Matrice de confusion d'un arbre de décision sur classes discontinues")
plt.show()

print("précision de l'arbre de décision : "+str(clf.score(X_test, y_test)))