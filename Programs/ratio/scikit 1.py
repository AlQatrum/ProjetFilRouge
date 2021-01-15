# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 15:51:51 2021

@author: Hello
"""

#importation des données

#importation des genres
import pandas as pd
import numpy as np
df=pd.read_csv('C:\\Users\\Hello\\Desktop\\IODAA\\fil_rouge\\to_aire.csv', sep=',', header=1 )
#df.reset_index(drop=True, inplace=True)
print (df.shape[1])
#df= df.drop([0])
df = df.drop(df.columns[0], axis=1)
df.columns = range(df.shape[1])
df= df.transpose()
print(df)

array_genus = df.values
array_genus = array_genus.astype(np.float)
print(array_genus)


#importation des ages
df=pd.read_csv('C:\\Users\\Hello\\Desktop\\IODAA\\fil_rouge\\to_age.csv', sep=',')
print (df.shape[1])
df = df.drop(df.columns[0], axis=1)
#df.columns = range(df.shape[1])
print(df)

array_age = df.values
array_age = array_age.astype(np.float)
print(array_age)




#array_age et array_ genus sont nos données pour scikit

#preparation des données

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(array_genus,array_age)


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30, 30,30))
mlp.fit(X_train, y_train)




